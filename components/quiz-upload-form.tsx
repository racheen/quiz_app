'use client';

import { useMemo, useState } from 'react';
import { useFormState, useFormStatus } from 'react-dom';
import { initialActionState } from '@/lib/action-state';
import { uploadQuizAction } from '@/lib/actions';

function SubmitButton() {
  const { pending } = useFormStatus();
  return (
    <button className="btn" type="submit" disabled={pending}>
      {pending ? 'Uploading...' : 'Upload Quiz JSON'}
    </button>
  );
}

const starterJson = `{
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
      "acceptedAnswers": ["training", "training set"],
      "explanation": "Overfitting usually means the model memorizes patterns from the training data."
    },
    {
      "type": "select_all",
      "prompt": "Select all supervised learning tasks.",
      "options": ["Classification", "Regression", "Clustering", "Dimensionality reduction"],
      "answerIndexes": [0, 1],
      "explanation": "Classification and regression use labeled data, while clustering and dimensionality reduction do not."
    }
  ]
}`;

export function QuizUploadForm({
  courses
}: {
  courses: { id: string; name: string; code: string }[];
}) {
  const [state, formAction] = useFormState(uploadQuizAction, initialActionState);
  const [jsonText, setJsonText] = useState(starterJson);

  const fileLabel = useMemo(() => 'Paste JSON below or choose a local .json file', []);

  return (
    <form action={formAction} className="form card">
      <div>
        <h3>Upload quiz</h3>
        <p className="muted">Each uploaded file creates one quiz under the selected course.</p>
      </div>
      <select className="select" name="courseId" required defaultValue="">
        <option value="" disabled>Select a course</option>
        {courses.map((course) => (
          <option key={course.id} value={course.id}>{course.code} — {course.name}</option>
        ))}
      </select>

      <label className="muted">{fileLabel}</label>
      <input
        className="input"
        type="file"
        accept="application/json"
        onChange={async (event) => {
          const file = event.target.files?.[0];
          if (!file) return;
          const text = await file.text();
          setJsonText(text);
        }}
      />

      <textarea
        className="textarea"
        name="jsonText"
        value={jsonText}
        onChange={(event) => setJsonText(event.target.value)}
      />
      <SubmitButton />
      {state.message ? <p className={state.ok ? 'success' : 'error'}>{state.message}</p> : null}
    </form>
  );
}
