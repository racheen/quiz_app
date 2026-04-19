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
DATABASE_URL="postgresql://postgres:postgres@localhost:5434/quiz_mvp?schema=public"
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

## Deploy on Vercel

Recommended MVP setup:

- Host the app on Vercel
- Host PostgreSQL on Neon

Deployment checklist:

1. Push this repo to GitHub.
2. Create a hosted Postgres database and copy its connection string into `DATABASE_URL` in Vercel.
3. Deploy the repo to Vercel.
4. Apply the Prisma schema to the hosted database:

```bash
npx prisma migrate deploy
```

5. Optional: seed starter data after the schema is live:

```bash
npm run db:seed
```

Notes:

- `postinstall` runs `prisma generate`, so Vercel builds the Prisma client automatically.
- This repo now includes an initial Prisma migration, so production should use `npx prisma migrate deploy`.
- `npx prisma db push` is still fine for quick local MVP work, but avoid it for long-term production schema history.

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
    },
    {
      "type": "select_all",
      "prompt": "Select all supervised learning tasks.",
      "options": [
        "Classification",
        "Regression",
        "Clustering",
        "Dimensionality reduction"
      ],
      "answerIndexes": [0, 1],
      "explanation": "Classification and regression use labeled data, while clustering and dimensionality reduction do not."
    },
    {
      "type": "ordering",
      "prompt": "Put the machine learning workflow in order.",
      "options": [
        "Train the model",
        "Collect data",
        "Evaluate the model",
        "Clean the data"
      ],
      "answerIndexes": [1, 3, 0, 2],
      "explanation": "A common flow is collect data, clean it, train the model, then evaluate it."
    },
    {
      "type": "multiple_choice",
      "prompt": "What is the output of:\n```python\ndef f():\n    return 10\n    print('Hi')\n\nprint(f())\n```",
      "options": [
        "Hi then 10",
        "10",
        "None",
        "Error"
      ],
      "answerIndex": 1,
      "explanation": "Code after return does not run, and print(f()) is outside the function, so the output is 10."
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
