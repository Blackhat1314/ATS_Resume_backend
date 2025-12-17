import express from 'express';
import cors from 'cors';
import multer from 'multer';
import pdfParse from 'pdf-parse';
import { GoogleGenerativeAI } from '@google/generative-ai';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import PDFDocument from 'pdfkit';
import { Readable } from 'stream';
import jwt from 'jsonwebtoken';
import rateLimit from 'express-rate-limit';
import mongoose from 'mongoose';
import bcrypt from 'bcryptjs';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3001;

// Environment variables - required for production
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const JWT_SECRET = process.env.JWT_SECRET || 'dev_jwt_secret_change_me';
const CORS_ORIGIN = process.env.CORS_ORIGIN || '*';

// MongoDB Configuration
const MONGODB_USERNAME = process.env.MONGODB_USERNAME;
const MONGODB_PASSWORD = process.env.MONGODB_PASSWORD;
const MONGODB_CLUSTER = process.env.MONGODB_CLUSTER;
const MONGODB_DATABASE = process.env.MONGODB_DATABASE || 'resume_ats_db';

// MongoDB Connection String
const MONGODB_URI = `mongodb+srv://${MONGODB_USERNAME}:${MONGODB_PASSWORD}@${MONGODB_CLUSTER}/${MONGODB_DATABASE}?retryWrites=true&w=majority&appName=Cluster0`;

// Validate required environment variables (only exit in production, allow fallback in development)
if (!GEMINI_API_KEY) {
  if (process.env.NODE_ENV === 'production') {
    console.error('ERROR: GEMINI_API_KEY is not set in environment variables!');
    console.error('Please create a .env file in the backend directory with GEMINI_API_KEY');
    process.exit(1);
  } else {
    console.warn('WARNING: GEMINI_API_KEY is not set. Please add it to backend/.env file');
  }
}

// Middleware
app.use(cors({
  origin: CORS_ORIGIN === '*' ? '*' : CORS_ORIGIN.split(',').map(origin => origin.trim()),
  credentials: true
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Rate limiting
const ipLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 150,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests from this IP, please try again later.' },
});

// User-based limiter (falls back to IP if unauthenticated)
const userLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 120,
  standardHeaders: true,
  legacyHeaders: false,
  keyGenerator: (req) => (req.user?.id || req.user?.email || req.ip),
  message: { error: 'Too many requests, please slow down.' },
});

app.use('/api', ipLimiter);

// User Schema
const userSchema = new mongoose.Schema({
  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true,
    trim: true
  },
  password: {
    type: String,
    required: true,
    minlength: 6
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

const User = mongoose.model('User', userSchema);

// Connect to MongoDB
if (MONGODB_USERNAME && MONGODB_PASSWORD && MONGODB_CLUSTER) {
  mongoose.connect(MONGODB_URI)
    .then(() => {
      console.log('✅ Connected to MongoDB successfully');
    })
    .catch((error) => {
      console.error('❌ MongoDB connection error:', error.message);
      if (process.env.NODE_ENV === 'production') {
        console.error('Fatal: Cannot connect to MongoDB in production');
        process.exit(1);
      } else {
        console.warn('Warning: MongoDB connection failed, but continuing in development mode');
      }
    });
} else {
  console.warn('⚠️  MongoDB credentials not found. Signup/Login will not work without MongoDB connection.');
}

// Auth helpers
function signToken(payload) {
  return jwt.sign(payload, JWT_SECRET, { expiresIn: '7d' });
}

function authenticateToken(req, res, next) {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  const token = authHeader.split(' ')[1];
  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded;
    next();
  } catch (err) {
    return res.status(401).json({ error: 'Invalid or expired token' });
  }
}

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(null, `resume-${Date.now()}.pdf`);
  }
});

const upload = multer({
  storage: storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'application/pdf') {
      cb(null, true);
    } else {
      cb(new Error('Only PDF files are allowed'), false);
    }
  }
});

// Initialize Gemini AI
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

// Retry function for API calls
async function retryApiCall(fn, maxRetries = 3, delay = 1000) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      console.log(`Attempt ${i + 1} failed:`, error.message);
      if (i === maxRetries - 1) {
        throw new Error(`API call failed after ${maxRetries} attempts: ${error.message}`);
      }
      await new Promise(resolve => setTimeout(resolve, delay * (i + 1)));
    }
  }
}

// Extract text from PDF
async function extractTextFromPDF(filePath) {
  try {
    const dataBuffer = fs.readFileSync(filePath);
    const data = await pdfParse(dataBuffer);
    return data.text;
  } catch (error) {
    throw new Error(`Failed to extract text from PDF: ${error.message}`);
  }
}

// Generate ATS analysis prompt
function generatePrompt(resumeText, jobDescription) {
  // Get current date for accurate experience calculation
  const now = new Date();
  const currentDate = now.toLocaleDateString('en-US', { month: 'long', year: 'numeric' }); // e.g., "December 2024"
  const currentYear = now.getFullYear();
  const currentMonth = now.getMonth() + 1; // 1-12
  const currentDay = now.getDate();
  
  return `You are an expert ATS (Applicant Tracking System) analyst. Analyze the following resume against the job description and provide a comprehensive assessment.

IMPORTANT: TODAY'S DATE IS ${currentDate} (Month: ${currentMonth}, Year: ${currentYear}, Day: ${currentDay}). Use this date when calculating experience for positions marked as "Currently", "Present", "Current", or any other indicator of ongoing employment.

CRITICAL: When analyzing the job description, IGNORE the following sections:
- Achievements section (any achievements, awards, or recognition mentioned)
- Soft skills section (communication, teamwork, leadership, etc. - focus only on technical/hard skills)
- About section (company description, culture, values, etc.)
- Focus ONLY on: Job requirements, technical skills, tools, technologies, experience requirements, responsibilities, and qualifications

RESUME TEXT:
${resumeText}

JOB DESCRIPTION:
${jobDescription}

TASKS TO PERFORM:

1. ATS MATCH SCORE
Give an overall ATS Match Score (0–100%), based on:
- Keyword match (technical keywords only)
- Skill match (TECHNICAL/HARD SKILLS ONLY - ignore soft skills like communication, teamwork, leadership)
- Experience relevance
- Tools, technologies, certifications
- Project alignment
- IGNORE: Soft skills, achievements, company culture/about sections from job description
Use strict ATS-style scoring focused on technical requirements.

2. KEYWORD ANALYSIS
- Matched Keywords
- Missing / Weak Keywords (that commonly appear in successful resumes for this JD)
Include ONLY:
- Technical skills (programming languages, frameworks, technologies)
- Tools & platforms (software, systems, applications)
- Domain-specific keywords (industry terms, methodologies)
- DO NOT include soft skills (communication, teamwork, leadership, etc.)
- DO NOT include achievements or awards from the job description
- Focus on hard skills and technical requirements only

3. ROLE FIT ANALYSIS
Explain:
- How well the resume matches the JD (focus on technical requirements and responsibilities)
- Which technical responsibilities are covered
- Which technical responsibilities are missing or weak
- IGNORE: Soft skill requirements, achievements, and "about company" sections when analyzing fit

4. EXPERIENCE EXTRACTION AND CALCULATION
CRITICAL: Before analyzing gaps, you MUST first extract and calculate experience correctly:

a) IDENTIFY ALL WORK EXPERIENCE:
- Look for sections titled: "Experience", "Work Experience", "Employment", "Professional Experience", "Career History"
- Also check for: "Internship", "Internships", "Intern Experience" (include these in total experience)
- Include ALL positions: full-time, part-time, contract, freelance, and internships

b) PARSE DATES CORRECTLY:
- Date formats vary. Look for patterns like:
  * "August 2023 – Currently" or "Aug 2023 – Present" (current role, calculate to TODAY'S DATE: ${currentDate})
  * "March 2022 – October 2022" or "Mar 2022 – Oct 2022" (past role)
  * "2020 – 2022" or "2020-2022" (year ranges)
  * "January 2021 – Present" or "Jan 2021 – Current" (current role, calculate to TODAY'S DATE: ${currentDate})
  * "08/2020 – 12/2022" (month/year format)
- Keywords indicating CURRENT role: "Currently", "Present", "Current", "Now", "Till date", "To date"
- CRITICAL: If end date shows "Currently/Present/Current", you MUST use TODAY'S DATE (${currentDate}, Month: ${currentMonth}, Year: ${currentYear}) as the end date for calculation
- Example calculation for "August 2023 – Currently":
  * Start Date: August 2023 (Month 8, Year 2023)
  * End Date: ${currentDate} (Month ${currentMonth}, Year ${currentYear}) - THIS IS TODAY'S DATE
  * Calculate months: From August 2023 to ${currentDate}
  * Convert to years: Divide total months by 12
  * Expected result: Approximately ${((currentYear - 2023) * 12 + (currentMonth >= 8 ? currentMonth - 8 : (12 - 8) + currentMonth)) / 12} years (NOT 0.8 years)
  * If you calculate 0.8 years for "August 2023 – Currently", you are WRONG. Recalculate using the current date provided above.

c) CALCULATE TOTAL YEARS:
- For EACH position, calculate duration in years (including partial years as decimals)
- Use TODAY'S DATE (${currentDate}, Month ${currentMonth}, Year ${currentYear}) when calculating current positions
- Calculation method (STEP BY STEP):
  1. Identify start month and year (e.g., August 2023 = Month 8, Year 2023)
  2. If end date is "Currently/Present", use TODAY: Month ${currentMonth}, Year ${currentYear}
  3. Calculate total months: (End Year - Start Year) × 12 + (End Month - Start Month)
  4. Convert to years: Total months ÷ 12
  5. Round to 1 decimal place
- Example: "August 2023 – Currently" 
  * Start: Month 8, Year 2023
  * End: Month ${currentMonth}, Year ${currentYear} (TODAY)
  * Months: (${currentYear} - 2023) × 12 + (${currentMonth} - 8) = ${(currentYear - 2023) * 12 + (currentMonth - 8)} months
  * Years: ${((currentYear - 2023) * 12 + (currentMonth - 8)) / 12} years (approximately ${Math.round(((currentYear - 2023) * 12 + (currentMonth - 8)) / 12 * 10) / 10} years)
  * THIS IS CORRECT - DO NOT use 0.8 years which is WRONG
- Example: "March 2022 – October 2022" = From Month 3, Year 2022 to Month 10, Year 2022 = 7 months = 0.6 years
- SUM ALL positions to get TOTAL years of experience
- Include internship experience in the total
- BE PRECISE: Always use the current date (${currentDate}) provided above for "Currently/Present" positions

d) IDENTIFY CURRENT ROLE:
- Mark which position is the current/active role
- Note the company name and job title of current role

5. GAP ANALYSIS
Based on the correctly calculated experience above, identify gaps in:
- Years of experience (State: "Candidate has X.X total years of experience [breakdown: Y years at Company A, Z years at Company B, etc.]. The job requires [requirement]. Gap: [difference]")
- Required tech stack (technical skills, tools, technologies)
- Domain knowledge (technical/industry knowledge)
- Certifications (technical certifications only)
- IGNORE gaps in: Soft skills, achievements, company culture fit

6. RESUME IMPROVEMENT SUGGESTIONS
Provide clear, actionable suggestions, such as:
- Missing TECHNICAL skills to add (ignore soft skills)
- Technical keywords to include
- How to rewrite bullets to show measurable impact
- What projects/experience can be better highlighted
- Formatting issues that reduce ATS score
- DO NOT suggest adding soft skills, achievements, or "about" content from the job description

7. IMPROVED RESUME BULLET POINTS
Rewrite 3–6 of the resume's weakest bullet points into strong, quantifiable, action-driven bullets following this structure:
Action Verb + Task + Method + Result (with metrics)

8. SPELLING AND GRAMMAR CHECK
Check the resume text for:
- Spelling errors (misspelled words)
- Grammar errors (incorrect grammar, punctuation, tense issues)
- Provide a list of all errors found with corrections

9. SKILLS COMPARISON
Extract all technical/hard skills mentioned in the job description and compare with the resume:
- For each skill in the job description, check if it appears in the resume
- Count how many times each skill appears in job description vs resume
- Mark skills as "required" if they appear in job description
- Create a comparison table with: Skill name, Job Description count, Resume count, Status (matched/missing)

10. FINAL OUTPUT FORMAT
Provide the response in the following JSON structure (use this exact format):
{
  "atsScore": 85,
  "overview": {
    "matchScore": 85,
    "summary": "Brief summary of how well the resume matches the job description, highlighting strengths and key gaps",
    "highlights": [
      "Key strength 1",
      "Key strength 2",
      "Key strength 3"
    ],
    "improvements": [
      "Key improvement 1",
      "Key improvement 2",
      "Key improvement 3"
    ],
    "radarScores": {
      "content": 85,
      "format": 75,
      "style": 80,
      "sections": 70,
      "skills": 90
    }
  },
  "spellingGrammar": {
    "errors": [
      {"type": "spelling", "word": "incorrect", "correction": "correct", "context": "sentence where error appears"},
      {"type": "grammar", "issue": "tense error", "correction": "corrected version", "context": "sentence where error appears"}
    ],
    "totalErrors": 2
  },
  "skillsComparison": [
    {"skill": "Node.js", "jobDescriptionCount": 6, "resumeCount": 4, "status": "matched", "required": true},
    {"skill": "Angular", "jobDescriptionCount": 5, "resumeCount": 0, "status": "missing", "required": true}
  ],
  "matchedKeywords": ["keyword1", "keyword2", "keyword3"],
  "missingKeywords": ["keyword1", "keyword2", "keyword3"],
  "responsibilityMatch": {
    "goodAreas": ["Area 1", "Area 2"],
    "missingAreas": ["Area 1", "Area 2"]
  },
  "gapAnalysis": {
    "experience": "Detailed analysis including: Total years calculated (X.X years), breakdown by position, current role identified, internship experience included, and gap compared to job requirements",
    "techStack": "Analysis of tech stack gaps",
    "domainKnowledge": "Analysis of domain knowledge gaps",
    "certifications": "Analysis of certification gaps",
    "achievements": "Analysis of achievement/metric gaps"
  },
  "experienceDetails": {
    "totalYears": 3.5,
    "currentRole": "Software Engineer at Company XYZ",
    "positions": [
      {"company": "Company A", "title": "Role", "duration": "2.5 years", "dates": "Jan 2020 - Jun 2022"},
      {"company": "Company B", "title": "Current Role", "duration": "1.0 years", "dates": "Jul 2022 - Present", "isCurrent": true}
    ],
    "includesInternships": true
  },
  "improvementSuggestions": [
    "Suggestion 1",
    "Suggestion 2",
    "Suggestion 3"
  ],
  "rewrittenBulletPoints": [
    "Improved bullet point 1",
    "Improved bullet point 2",
    "Improved bullet point 3"
  ],
  "finalVerdict": "Should the candidate apply? Provide a clear recommendation with reasoning."
}

IMPORTANT: Return ONLY valid JSON. Do not include any markdown formatting, code blocks, or additional text outside the JSON object.`;
}

// Auth endpoints
// Signup endpoint
app.post('/api/auth/signup', async (req, res) => {
  try {
    const { email, password } = req.body;
    
    if (!email || !password) {
      return res.status(400).json({ error: 'Email and password are required' });
    }

    if (password.length < 6) {
      return res.status(400).json({ error: 'Password must be at least 6 characters long' });
    }

    // Check if user already exists
    const existingUser = await User.findOne({ email: email.toLowerCase() });
    if (existingUser) {
      return res.status(400).json({ error: 'User with this email already exists' });
    }

    // Hash password
    const saltRounds = 10;
    const hashedPassword = await bcrypt.hash(password, saltRounds);

    // Create new user
    const user = new User({
      email: email.toLowerCase(),
      password: hashedPassword
    });

    await user.save();

    // Generate token
    const token = signToken({ email: user.email, id: user._id.toString() });

    res.status(201).json({
      token,
      user: { email: user.email, id: user._id.toString() },
      message: 'User created successfully'
    });
  } catch (error) {
    console.error('Signup error:', error);
    if (error.code === 11000) {
      return res.status(400).json({ error: 'User with this email already exists' });
    }
    res.status(500).json({ error: 'Failed to create user', message: error.message });
  }
});

// Login endpoint
app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    
    if (!email || !password) {
      return res.status(400).json({ error: 'Email and password are required' });
    }

    // Find user by email
    const user = await User.findOne({ email: email.toLowerCase() });
    if (!user) {
      return res.status(401).json({ error: 'Invalid email or password' });
    }

    // Verify password
    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) {
      return res.status(401).json({ error: 'Invalid email or password' });
    }

    // Generate token
    const token = signToken({ email: user.email, id: user._id.toString() });

    res.json({
      token,
      user: { email: user.email, id: user._id.toString() }
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'Failed to login', message: error.message });
  }
});


// API endpoint for resume analysis
app.post('/api/analyze-resume', authenticateToken, userLimiter, upload.single('resume'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No PDF file uploaded' });
    }

    if (!req.body.jobDescription || req.body.jobDescription.trim() === '') {
      // Clean up uploaded file
      fs.unlinkSync(req.file.path);
      return res.status(400).json({ error: 'Job description is required' });
    }

    // Extract text from PDF
    const resumeText = await extractTextFromPDF(req.file.path);
    
    if (!resumeText || resumeText.trim() === '') {
      fs.unlinkSync(req.file.path);
      return res.status(400).json({ error: 'Could not extract text from PDF. Please ensure the PDF contains readable text.' });
    }

    const jobDescription = req.body.jobDescription;

    // Generate prompt
    const prompt = generatePrompt(resumeText, jobDescription);

    // Call Gemini API with retry logic
    const analysisResult = await retryApiCall(async () => {
      // Try gemini-2.5-pro first, fallback to gemini-2.5-flash
      const modelNames = ['gemini-2.5-flash'];

      // 'gemini-2.5-pro'
      
      let lastError = null;
      
      for (const modelName of modelNames) {
        try {
          console.log(`Trying model: ${modelName}`);
          const model = genAI.getGenerativeModel({ model: modelName });
          const response = await model.generateContent(prompt);
          // Use the correct response structure: response.response.text() (double response)
          const text = response.response.text();
          
          // If we get here, the model worked
          lastError = null;
          
          // Try to parse JSON from response
          let jsonText = text.trim();
          
          // Remove markdown code blocks if present
          if (jsonText.startsWith('```json')) {
            jsonText = jsonText.replace(/```json\n?/g, '').replace(/```\n?/g, '');
          } else if (jsonText.startsWith('```')) {
            jsonText = jsonText.replace(/```\n?/g, '');
          }
          
          try {
            return JSON.parse(jsonText);
          } catch (parseError) {
            // If JSON parsing fails, try to extract JSON from the text
            const jsonMatch = jsonText.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
              return JSON.parse(jsonMatch[0]);
            }
            throw new Error('Failed to parse JSON response from AI');
          }
        } catch (error) {
          console.log(`Model ${modelName} failed:`, error.message);
          lastError = error;
          // Continue to next model if it's a 404 or "not found" error
          if (error.message.includes('not found') || error.message.includes('404') || error.message.includes('overloaded')) {
            if (modelName === modelNames[modelNames.length - 1]) {
              // Last model failed, throw error
              throw new Error(`All model attempts failed. Last error: ${error.message}`);
            }
            continue;
          } else {
            // For other errors, throw immediately
            throw error;
          }
        }
      }
      
      // If all models failed, throw the last error
      if (lastError) {
        throw new Error(`All model attempts failed. Last error: ${lastError.message}`);
      }
      
      throw new Error('No models available');
      
      // Try to parse JSON from response
      let jsonText = text.trim();
      
      // Remove markdown code blocks if present
      if (jsonText.startsWith('```json')) {
        jsonText = jsonText.replace(/```json\n?/g, '').replace(/```\n?/g, '');
      } else if (jsonText.startsWith('```')) {
        jsonText = jsonText.replace(/```\n?/g, '');
      }
      
      try {
        return JSON.parse(jsonText);
      } catch (parseError) {
        // If JSON parsing fails, try to extract JSON from the text
        const jsonMatch = jsonText.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          return JSON.parse(jsonMatch[0]);
        }
        throw new Error('Failed to parse JSON response from AI');
      }
    });

    // Store file info for potential editing (don't delete yet)
    const fileId = `file-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const tempFilesDir = path.join(__dirname, 'temp_files');
    if (!fs.existsSync(tempFilesDir)) {
      fs.mkdirSync(tempFilesDir, { recursive: true });
    }
    
    // Move file to temp directory with ID
    const tempFilePath = path.join(tempFilesDir, `${fileId}.pdf`);
    fs.copyFileSync(req.file.path, tempFilePath);
    
    // Store resume text with file ID
    const resumeDataPath = path.join(tempFilesDir, `${fileId}.txt`);
    fs.writeFileSync(resumeDataPath, resumeText);
    
    // Clean up original uploaded file
    fs.unlinkSync(req.file.path);
    
    // Schedule cleanup after 1 hour
    setTimeout(() => {
      if (fs.existsSync(tempFilePath)) fs.unlinkSync(tempFilePath);
      if (fs.existsSync(resumeDataPath)) fs.unlinkSync(resumeDataPath);
    }, 3600000);

    // Return the analysis result with file ID for editing
    res.json({
      success: true,
      data: {
        ...analysisResult,
        fileId: fileId // Include file ID for potential editing
      }
    });

  } catch (error) {
    console.error('Error analyzing resume:', error);
    
    // Clean up uploaded file if it exists
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
    
    res.status(500).json({
      error: 'Failed to analyze resume',
      message: error.message
    });
  }
});

// Mock questions generation endpoint
app.post('/api/mock-questions', authenticateToken, userLimiter, upload.single('resume'), async (req, res) => {
  console.log('[MOCK QUESTIONS] Request received');
  try {
    if (!req.file) {
      console.log('[MOCK QUESTIONS] Error: No PDF file uploaded');
      return res.status(400).json({ error: 'No PDF file uploaded' });
    }
    console.log('[MOCK QUESTIONS] PDF file received:', req.file.filename);
    
    if (!req.body.jobDescription || req.body.jobDescription.trim() === '') {
      console.log('[MOCK QUESTIONS] Error: Job description missing');
      fs.unlinkSync(req.file.path);
      return res.status(400).json({ error: 'Job description is required' });
    }
    console.log('[MOCK QUESTIONS] Job description received, length:', req.body.jobDescription.length);

    console.log('[MOCK QUESTIONS] Extracting text from PDF...');
    const resumeText = await extractTextFromPDF(req.file.path);
    if (!resumeText || resumeText.trim() === '') {
      console.log('[MOCK QUESTIONS] Error: Could not extract text from PDF');
      fs.unlinkSync(req.file.path);
      return res.status(400).json({ error: 'Could not extract text from PDF. Please ensure the PDF contains readable text.' });
    }
    console.log('[MOCK QUESTIONS] Resume text extracted, length:', resumeText.length);

    const jobDescription = req.body.jobDescription;
    const manualYearsOfExperience = parseFloat(req.body.yearsOfExperience) || 0;
    let companyName = (req.body.companyName || '').trim();
    
    console.log('[MOCK QUESTIONS] Generating prompt...');
    console.log(`[MOCK QUESTIONS] Using manual years of experience: ${manualYearsOfExperience}`);
    console.log(`[MOCK QUESTIONS] Company name: ${companyName || 'Not provided'}`);
    
    if (!manualYearsOfExperience || manualYearsOfExperience <= 0) {
      fs.unlinkSync(req.file.path);
      return res.status(400).json({ error: 'Valid years of experience is required' });
    }

    // Try to extract company name from job description if not provided
    if (!companyName) {
      console.log('[MOCK QUESTIONS] Attempting to extract company name from job description...');
      const companyExtractionPrompt = `Extract the company name from this job description. Return ONLY the company name, nothing else. If no company name is found, return "NONE".

JOB DESCRIPTION:
${jobDescription}`;

      try {
        const companyModel = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
        const companyResponse = await companyModel.generateContent(companyExtractionPrompt);
        const extractedCompany = companyResponse.response.text().trim().toUpperCase();
        if (extractedCompany && extractedCompany !== 'NONE' && extractedCompany.length < 100) {
          companyName = extractedCompany;
          console.log(`[MOCK QUESTIONS] Extracted company name: ${companyName}`);
        }
      } catch (err) {
        console.log('[MOCK QUESTIONS] Could not extract company name from job description');
      }
    }

    // Generate questions in three categories with difficulty levels
    const prompt = `You are an expert interviewer. Generate mock interview questions for this candidate based on their resume and the job description.

RESUME:
${resumeText}

JOB DESCRIPTION:
${jobDescription}

CANDIDATE'S YEARS OF EXPERIENCE: ${manualYearsOfExperience} years

Generate questions in THREE categories:

1. JOB DESCRIPTION BASED QUESTIONS (45 questions total):
   - Questions based on the job requirements, responsibilities, and skills mentioned in the job description
   - Categorize by difficulty based on the candidate's ${manualYearsOfExperience} years of experience:
     * EASY: Basic questions suitable for someone with ${manualYearsOfExperience} years of experience (15 questions)
     * MEDIUM: Intermediate questions that challenge someone with ${manualYearsOfExperience} years of experience (15 questions)
     * HARD: Advanced questions that would be challenging even for someone with ${manualYearsOfExperience} years of experience (15 questions)

2. RESUME BASED QUESTIONS (45 questions total):
   - Questions based on the candidate's actual experience, projects, and skills mentioned in their resume
   - Categorize by difficulty based on the candidate's ${manualYearsOfExperience} years of experience:
     * EASY: Basic questions about their experience (15 questions)
     * MEDIUM: Intermediate questions that dig deeper into their experience (15 questions)
     * HARD: Advanced questions that test deep understanding of their work (15 questions)

3. DSA AND SYSTEM DESIGN QUESTIONS (3 questions total):
   CRITICAL INSTRUCTIONS FOR DSA QUESTIONS:
   - DO NOT create or generate DSA questions yourself
   - You MUST search for and provide REAL, ACTUAL LeetCode problems that exist on leetcode.com
   ${companyName ? `- Search for LeetCode problems that are commonly asked in ${companyName} interviews
   - Use your knowledge of ${companyName} interview patterns and frequently asked LeetCode problems
   - Examples: Google often asks problems like "Two Sum", "LRU Cache", "Merge Intervals"
   - Examples: Amazon often asks problems like "Two Sum", "Longest Palindromic Substring", "Trapping Rain Water"
   - Examples: Microsoft often asks problems like "Reverse Linked List", "Valid Parentheses", "Best Time to Buy and Sell Stock"
   - Provide 2 REAL LeetCode problems that match ${companyName}'s interview style and the candidate's ${manualYearsOfExperience} years of experience` : `- Search for LeetCode problems appropriate for ${manualYearsOfExperience} years of experience
   - For ${manualYearsOfExperience <= 2 ? 'junior level (Easy to Medium)' : manualYearsOfExperience <= 5 ? 'mid-level (Medium to Hard)' : 'senior level (Hard)'} candidates
   - Provide 2 REAL LeetCode problems matching this experience level`}
   - Format each DSA question as: "LeetCode [Problem Number]: [Problem Title] - [Brief 1-sentence description]"
   - Include the actual LeetCode problem number if you know it (e.g., "LeetCode 1: Two Sum - Find two numbers that add up to target")
   - If you don't know the exact number, use: "LeetCode: [Problem Title] - [Brief description]"
   - These must be REAL problems that exist on LeetCode, not made-up questions
   
   - Generate 1 System Design question (you can create this, but make it realistic)
   - System Design difficulty based on ${manualYearsOfExperience} years:
     * ${manualYearsOfExperience <= 2 ? 'Basic system design (e.g., Design a URL shortener, Design a chat system)' : manualYearsOfExperience <= 5 ? 'Scalable system design (e.g., Design Twitter, Design a distributed cache)' : 'Large-scale distributed systems (e.g., Design YouTube, Design a global CDN)'}

Rules:
- Generate exactly 15 questions per difficulty level per category (total: 90 questions for first two categories)
- Generate exactly 2 DSA questions and 1 System Design question (total: 3 questions for third category)
- Tailor difficulty to the candidate's ${manualYearsOfExperience} years of experience
- Focus on technical depth, system design, frameworks, and tools
- Include scenario-based questions, debugging questions, and performance optimization where relevant
- Avoid generic soft-skill questions
- Return ONLY valid JSON in this exact format (no markdown, no code blocks):

{
  "jobDescriptionBased": {
    "easy": ["question 1", "question 2", ..., "question 15"],
    "medium": ["question 1", "question 2", ..., "question 15"],
    "hard": ["question 1", "question 2", ..., "question 15"]
  },
  "resumeBased": {
    "easy": ["question 1", "question 2", ..., "question 15"],
    "medium": ["question 1", "question 2", ..., "question 15"],
    "hard": ["question 1", "question 2", ..., "question 15"]
  },
  "dsaAndSystemDesign": {
    "dsa": ["DSA question 1", "DSA question 2"],
    "systemDesign": ["System Design question 1"]
  }
}`;

    console.log('[MOCK QUESTIONS] Calling AI to generate questions...');
    const questionsData = await retryApiCall(async () => {
      const modelNames = ['gemini-2.5-flash'];
      let lastError = null;
      for (const modelName of modelNames) {
        try {
          console.log(`[MOCK QUESTIONS] Trying model: ${modelName}`);
          const model = genAI.getGenerativeModel({ model: modelName });
          const response = await model.generateContent(prompt);
          let text = response.response.text().trim();
          console.log('[MOCK QUESTIONS] AI response received, length:', text.length);
          
          // Strip markdown fences if any
          if (text.startsWith('```')) {
            text = text.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
          }
          
          try {
            const parsed = JSON.parse(text);
            if (parsed.jobDescriptionBased && parsed.resumeBased && parsed.dsaAndSystemDesign) {
              console.log(`[MOCK QUESTIONS] Successfully parsed questions from JSON`);
              return parsed;
            } else {
              console.log('[MOCK QUESTIONS] Missing required sections in response');
            }
          } catch (err) {
            console.log('[MOCK QUESTIONS] JSON parse failed:', err.message);
          }
          throw new Error('Failed to parse questions from AI response');
        } catch (err) {
          console.log(`[MOCK QUESTIONS] Model ${modelName} failed:`, err.message);
          lastError = err;
          if (modelName === modelNames[modelNames.length - 1]) throw err;
        }
      }
      throw lastError;
    });

    console.log(`[MOCK QUESTIONS] Generated questions successfully`);
    fs.unlinkSync(req.file.path);
    res.json({ 
      success: true, 
      data: { 
        questions: {
          ...questionsData,
          companyName: companyName || null
        },
        yearsOfExperience: manualYearsOfExperience
      } 
    });
  } catch (error) {
    console.error('[MOCK QUESTIONS] Error:', error.message);
    console.error('[MOCK QUESTIONS] Stack:', error.stack);
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
    res.status(500).json({ error: 'Failed to generate mock questions', message: error.message });
  }
});

// Generate improved resume PDF endpoint
app.post('/api/generate-improved-resume', authenticateToken, userLimiter, async (req, res) => {
  try {
    const { fileId, missingKeywords, improvementSuggestions, jobDescription } = req.body;

    if (!fileId) {
      return res.status(400).json({ error: 'File ID is required' });
    }

    const tempFilesDir = path.join(__dirname, 'temp_files');
    if (!fs.existsSync(tempFilesDir)) {
      fs.mkdirSync(tempFilesDir, { recursive: true });
    }
    
    const resumeDataPath = path.join(tempFilesDir, `${fileId}.txt`);
    const originalPdfPath = path.join(tempFilesDir, `${fileId}.pdf`);

    if (!fs.existsSync(resumeDataPath)) {
      return res.status(404).json({ error: 'Resume data not found. Please re-upload your resume.' });
    }

    // Read original resume text
    const originalResumeText = fs.readFileSync(resumeDataPath, 'utf-8');

    // Generate improved resume using AI
    const improvementPrompt = `You are an expert resume writer. Your task is to IMPROVE the existing resume by adding missing keywords and enhancements while PRESERVING THE EXACT FORMAT AND STRUCTURE.

MISSING KEYWORDS TO ADD: ${missingKeywords?.join(', ') || 'N/A'}

IMPROVEMENT SUGGESTIONS: ${improvementSuggestions?.join('; ') || 'N/A'}

JOB DESCRIPTION: ${jobDescription}

ORIGINAL RESUME TEXT:
${originalResumeText}

CRITICAL INSTRUCTIONS:
1. PRESERVE THE EXACT STRUCTURE: Keep all section headers, formatting, and layout exactly as they appear in the original
2. ADD KEYWORDS NATURALLY: Integrate missing keywords into existing bullet points and descriptions without changing the meaning
3. ENHANCE EXISTING CONTENT: Improve bullet points by adding metrics and quantifiable results, but keep the same structure
4. ADD TO SKILLS SECTION: If there's a skills section, add missing technical skills there
5. MAINTAIN FORMATTING: Keep the same line breaks, spacing, and section organization
6. DO NOT REMOVE: Keep all original experience, education, and achievements
7. DO NOT RESTRUCTURE: Do not reorganize sections or change the order
8. SMART ADDITIONS: Add missing keywords in context - for example, if "Python" is missing and the person has backend experience, add it to relevant bullet points

OUTPUT FORMAT:
- Return the improved resume text with the EXACT same structure as the original
- Use the same section headers (EXPERIENCE, EDUCATION, SKILLS, etc.)
- Maintain the same formatting style
- Keep bullet points in the same format (• or -)
- Preserve dates and company names exactly as they appear
- Only add/enhance content, do not remove or restructure

Return ONLY the improved resume text. Do not include explanations, markdown, or code blocks.`;

    const improvedResumeText = await retryApiCall(async () => {
      const modelNames = ['gemini-2.5-pro', 'gemini-2.5-flash'];
      
      for (const modelName of modelNames) {
        try {
          console.log(`Generating improved resume with model: ${modelName}`);
          const model = genAI.getGenerativeModel({ model: modelName });
          const response = await model.generateContent(improvementPrompt);
          return response.response.text();
        } catch (error) {
          if (modelName === modelNames[modelNames.length - 1]) {
            throw error;
          }
          continue;
        }
      }
    });

    // Create a new PDF with improved resume text using PDFKit (better text layout)
    const pdfBuffer = await new Promise((resolve, reject) => {
      try {
        const doc = new PDFDocument({
          size: 'LETTER',
          margin: 50
        });

        const buffers = [];
        doc.on('data', buffers.push.bind(buffers));
        doc.on('end', () => resolve(Buffer.concat(buffers)));
        doc.on('error', reject);

        // Parse and format the improved resume text
        const lines = improvedResumeText.split('\n');
        let isFirstLine = true;

        for (let i = 0; i < lines.length; i++) {
          const line = lines[i].trim();
          
          // Skip empty lines but add spacing
          if (!line) {
            doc.moveDown(0.5);
            continue;
          }

          // Check if line is a heading (all caps, short, no periods, or common section headers)
          const isHeading = (
            line === line.toUpperCase() && 
            line.length < 60 && 
            !line.includes('.') &&
            (line.length > 3 || ['EDUCATION', 'EXPERIENCE', 'SKILLS', 'PROJECTS', 'CERTIFICATIONS', 'AWARDS'].some(h => line.includes(h)))
          ) || 
          ['EDUCATION', 'EXPERIENCE', 'WORK EXPERIENCE', 'PROFESSIONAL EXPERIENCE', 
           'SKILLS', 'TECHNICAL SKILLS', 'PROJECTS', 'CERTIFICATIONS', 'AWARDS', 
           'SUMMARY', 'OBJECTIVE', 'CONTACT', 'PROFILE'].some(header => 
            line.toUpperCase().includes(header) && line.length < 60
          );

          if (isHeading) {
            // Add space before heading (except first)
            if (!isFirstLine) {
              doc.moveDown(1);
            }
            // Heading style
            doc.fontSize(14)
               .font('Helvetica-Bold')
               .text(line, {
                 align: 'left',
                 paragraphGap: 5
               });
            doc.fontSize(10)
               .font('Helvetica');
          } else {
            // Regular text - check if it's a bullet point
            if (line.startsWith('•') || line.startsWith('-') || line.startsWith('*') || /^\d+\./.test(line)) {
              // Bullet point - pdfkit handles text wrapping automatically
              doc.fontSize(10)
                 .font('Helvetica')
                 .text(line, {
                   align: 'left',
                   indent: 20,
                   paragraphGap: 3,
                   lineGap: 2,
                   width: doc.page.width - 100 - 20 // Account for margin and indent
                 });
            } else {
              // Regular paragraph text - pdfkit handles text wrapping automatically
              doc.fontSize(10)
                 .font('Helvetica')
                 .text(line, {
                   align: 'left',
                   paragraphGap: 5,
                   lineGap: 2,
                   width: doc.page.width - 100 // Account for margins
                 });
            }
          }

          isFirstLine = false;
        }

        doc.end();
      } catch (error) {
        reject(error);
      }
    });

    // Send PDF as response
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', 'attachment; filename="improved-resume.pdf"');
    res.send(pdfBuffer);

  } catch (error) {
    console.error('Error generating improved resume:', error);
    res.status(500).json({
      error: 'Failed to generate improved resume',
      message: error.message
    });
  }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'Server is running' });
});

// Start server
app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server is running on ${PORT}`);
});




