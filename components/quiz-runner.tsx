'use client';

import { useEffect, useMemo, useRef, useState } from 'react';

type Question = {
  id: string;
  type: 'multiple_choice' | 'fill_blank';
  prompt: string;
  options: string[];
  answerIndex: number | null;
  acceptedAnswers: string[];
  explanation: string | null;
  order: number;
};

function normalizeFillBlankAnswer(value: string) {
  return value.trim().toLowerCase().replace(/\s+/g, ' ');
}

export function QuizRunner({ quizTitle, questions }: { quizTitle: string; questions: Question[] }) {
  const [answers, setAnswers] = useState<Record<string, string | number>>({});
  const [submitted, setSubmitted] = useState(false);
  const summaryRef = useRef<HTMLDivElement | null>(null);

  const score = useMemo(() => {
    return questions.reduce((total, question) => {
      const answer = answers[question.id];

      if (question.type === 'fill_blank') {
        if (typeof answer !== 'string') {
          return total;
        }

        const normalizedAnswer = normalizeFillBlankAnswer(answer);
        const isCorrect = question.acceptedAnswers.some(
          (acceptedAnswer) => normalizeFillBlankAnswer(acceptedAnswer) === normalizedAnswer
        );

        return total + (isCorrect ? 1 : 0);
      }

      return total + (typeof answer === 'number' && answer === question.answerIndex ? 1 : 0);
    }, 0);
  }, [answers, questions]);

  const answeredCount = Object.keys(answers).length;
  const unansweredCount = questions.length - answeredCount;
  const percentage = questions.length === 0 ? 0 : Math.round((score / questions.length) * 100);

  useEffect(() => {
    if (submitted) {
      summaryRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [submitted]);

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="card quiz-header" ref={summaryRef}>
        <h1>{quizTitle}</h1>
        <p className="muted">
          {submitted
            ? 'Your results are ready below. Review each question to see what was correct and any explanations provided.'
            : 'Choose one answer for each question, then submit to see your score and review.'}
        </p>
        {submitted ? (
          <div className="quiz-summary">
            <div className="quiz-score-ring" aria-label={`Score ${score} out of ${questions.length}`}>
              <strong>{percentage}%</strong>
              <span>{score} / {questions.length}</span>
            </div>
            <div className="quiz-summary-copy">
              <h2>Quiz complete</h2>
              <p className="muted">
                You answered <strong>{answeredCount}</strong> question{answeredCount === 1 ? '' : 's'} and got{' '}
                <strong>{score}</strong> correct.
              </p>
              {unansweredCount > 0 ? (
                <p className="muted">
                  <strong>{unansweredCount}</strong> question{unansweredCount === 1 ? ' was' : 's were'} left unanswered.
                </p>
              ) : null}
            </div>
          </div>
        ) : (
          <p className="muted">Answered {answeredCount} of {questions.length}</p>
        )}
      </div>

      {questions.map((question, idx) => {
        const selected = answers[question.id];
        const wasAnswered =
          question.type === 'fill_blank'
            ? typeof selected === 'string' && selected.trim().length > 0
            : typeof selected === 'number';
        const isCorrect =
          question.type === 'fill_blank'
            ? typeof selected === 'string' &&
              question.acceptedAnswers.some(
                (acceptedAnswer) =>
                  normalizeFillBlankAnswer(acceptedAnswer) === normalizeFillBlankAnswer(selected)
              )
            : typeof selected === 'number' && selected === question.answerIndex;
        const selectedLabel =
          question.type === 'fill_blank'
            ? typeof selected === 'string'
              ? selected
              : null
            : typeof selected === 'number'
              ? question.options[selected]
              : null;
        const correctLabel =
          question.type === 'fill_blank'
            ? question.acceptedAnswers.join(', ')
            : question.answerIndex !== null
              ? question.options[question.answerIndex]
              : '';
        return (
          <div className="card" key={question.id}>
            <h3>{idx + 1}. {question.prompt}</h3>
            {question.type === 'fill_blank' ? (
              <input
                className="input"
                type="text"
                name={question.id}
                value={typeof selected === 'string' ? selected : ''}
                placeholder="Type your answer"
                onChange={(event) => setAnswers((prev) => ({ ...prev, [question.id]: event.target.value }))}
              />
            ) : (
              <div className="list">
                {question.options.map((option, optionIndex) => {
                  const optionIsCorrect = submitted && optionIndex === question.answerIndex;
                  const optionIsWrong = submitted && selected === optionIndex && optionIndex !== question.answerIndex;
                  return (
                    <label key={optionIndex} className={`option ${optionIsCorrect ? 'correct' : ''} ${optionIsWrong ? 'wrong' : ''}`}>
                      <input
                        type="radio"
                        name={question.id}
                        checked={selected === optionIndex}
                        onChange={() => setAnswers((prev) => ({ ...prev, [question.id]: optionIndex }))}
                      />
                      <span>{option}</span>
                    </label>
                  );
                })}
              </div>
            )}
            {submitted ? (
              <div className="quiz-feedback">
                <p className={isCorrect ? 'success' : 'error'}>
                  {isCorrect
                    ? 'Correct.'
                    : wasAnswered
                      ? 'Not quite.'
                      : 'No answer selected.'}
                </p>
                {wasAnswered ? (
                  <p className="muted"><strong>Your answer:</strong> {selectedLabel}</p>
                ) : null}
                <p className="muted"><strong>Correct answer:</strong> {correctLabel}</p>
                {question.explanation ? (
                  <div className="quiz-explanation">
                    <strong>Explanation</strong>
                    <p className="muted">{question.explanation}</p>
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>
        );
      })}

      <div>
        <button className="btn" onClick={() => setSubmitted(true)}>
          {submitted ? 'Review Complete' : 'Submit Quiz'}
        </button>
      </div>
    </div>
  );
}
