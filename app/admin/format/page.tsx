const exampleJson = {
  title: 'Week 9-12 Mock Exam 1',
  slug: 'week-9-12-mock-exam-1',
  description: 'Practice quiz with mixed question types',
  questions: [
    {
      type: 'multiple_choice',
      prompt: 'What does NLP stand for?',
      options: [
        'Natural Language Processing',
        'Neural Logic Protocol',
        'Numeric Language Parsing',
        'Natural Learning Program'
      ],
      answerIndex: 0,
      explanation: 'NLP stands for Natural Language Processing.'
    },
    {
      type: 'fill_blank',
      prompt: 'Fill in the blank: Overfitting happens when a model learns the ______ data too closely.',
      acceptedAnswers: ['training', 'training set'],
      explanation: 'Overfitting usually means the model memorizes patterns from the training data.'
    }
  ]
};

export default function JsonFormatPage() {
  return (
    <main className="container grid" style={{ gap: 20 }}>
      <div className="card">
        <h1>Quiz JSON format</h1>
        <p className="muted">
          Upload one JSON object per quiz. The selected course on the admin page decides where the quiz is stored.
        </p>
      </div>

      <div className="card">
        <h3>Required structure</h3>
        <pre>{JSON.stringify(exampleJson, null, 2)}</pre>
      </div>

      <div className="card">
        <h3>Rules</h3>
        <ul>
          <li><code>title</code> is required.</li>
          <li><code>slug</code> is optional. If omitted, it is generated from the title.</li>
          <li><code>questions</code> must contain at least one question.</li>
          <li>Use <code>type: "multiple_choice"</code> with <code>options</code> and <code>answerIndex</code> for standard MCQs.</li>
          <li>Use <code>type: "fill_blank"</code> with <code>acceptedAnswers</code> for text-entry questions.</li>
          <li><code>answerIndex</code> must point to a valid option for multiple-choice questions.</li>
          <li><code>acceptedAnswers</code> accepts one or more valid answers, and grading ignores case and extra spaces.</li>
        </ul>
      </div>
    </main>
  );
}
