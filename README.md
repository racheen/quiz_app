# Quiz MVP Postgres

A focused MVP for course-based quiz uploads using **Next.js + Prisma + PostgreSQL**.

## Features

- Add courses
- Upload quiz JSON into a selected course
- Support multiple quizzes per course
- Take quizzes and get an instant score
- Run Postgres locally with Docker

## Stack

- Next.js 14 App Router
- Prisma ORM
- PostgreSQL
- Docker Compose

## 1) Start Postgres locally

```bash
docker compose up -d
```

## 2) Configure env

Copy the example env file:

```bash
cp .env.example .env
```

Default local connection string:

```env
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/quiz_mvp?schema=public"
```

## 3) Install dependencies

```bash
npm install
```

## 4) Create the database schema

```bash
npm run db:generate
npm run db:push
```

Or use migrations instead:

```bash
npm run db:migrate
```

## 5) Optional seed data

```bash
npm run db:seed
```

## 6) Run the app

```bash
npm run dev
```

Open `http://localhost:3000`

## JSON upload format

Upload one quiz JSON at a time.

```json
{
  "title": "Week 9-12 Mock Exam 1",
  "slug": "week-9-12-mock-exam-1",
  "description": "Practice quiz with mixed question types",
  "questions": [
    {
      "type": "multiple_choice",
      "prompt": "What does NLP stand for?",
      "options": [
        "Natural Language Processing",
        "Neural Logic Protocol",
        "Numeric Language Parsing",
        "Natural Learning Program"
      ],
      "answerIndex": 0,
      "explanation": "NLP stands for Natural Language Processing."
    },
    {
      "type": "fill_blank",
      "prompt": "Fill in the blank: Overfitting happens when a model learns the ______ data too closely.",
      "acceptedAnswers": [
        "training",
        "training set"
      ],
      "explanation": "Overfitting usually means the model memorizes patterns from the training data."
    }
  ]
}
```

## Data model

- `Course`
- `Quiz`
- `Question`

Each quiz belongs to one course, and each course can have many quizzes.

## Suggested next upgrades

- Admin auth
- Edit/delete course and quiz
- Save quiz attempts and scores
- Timer mode
- Randomized question order
- CSV export of attempts
