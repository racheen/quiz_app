'use client';

import { useEffect, useMemo, useRef, useState } from 'react';

type Question = {
  id: string;
  type: 'multiple_choice' | 'select_all' | 'fill_blank';
  prompt: string;
  options: string[];
  answerIndex: number | null;
  answerIndexes: number[];
  acceptedAnswers: string[];
  explanation: string | null;
  order: number;
};

type AnswerValue = string | number | number[];

function normalizeFillBlankAnswer(value: string) {
  return value.trim().toLowerCase().replace(/\s+/g, ' ');
}

function normalizeSelectedIndexes(value: number[]) {
  return [...value].sort((a, b) => a - b);
}

function answersMatch(selected: number[], expected: number[]) {
  const normalizedSelected = normalizeSelectedIndexes(selected);
  const normalizedExpected = normalizeSelectedIndexes(expected);

  return (
    normalizedSelected.length === normalizedExpected.length &&
    normalizedSelected.every((value, index) => value === normalizedExpected[index])
  );
}

export function QuizRunner({ quizTitle, questions }: { quizTitle: string; questions: Question[] }) {
  const [answers, setAnswers] = useState<Record<string, AnswerValue>>({});
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

      if (question.type === 'select_all') {
        return total + (Array.isArray(answer) && answersMatch(answer, question.answerIndexes) ? 1 : 0);
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
            : 'Answer each question, then submit to see your score and review.'}
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
            : question.type === 'select_all'
              ? Array.isArray(selected) && selected.length > 0
              : typeof selected === 'number';
        const isCorrect =
          question.type === 'fill_blank'
            ? typeof selected === 'string' &&
              question.acceptedAnswers.some(
                (acceptedAnswer) =>
                  normalizeFillBlankAnswer(acceptedAnswer) === normalizeFillBlankAnswer(selected)
              )
            : question.type === 'select_all'
              ? Array.isArray(selected) && answersMatch(selected, question.answerIndexes)
              : typeof selected === 'number' && selected === question.answerIndex;
        const selectedLabel =
          question.type === 'fill_blank'
            ? typeof selected === 'string'
              ? selected
              : null
            : question.type === 'select_all'
              ? Array.isArray(selected) && selected.length > 0
                ? normalizeSelectedIndexes(selected).map((answerIndex) => question.options[answerIndex]).join(', ')
                : null
              : typeof selected === 'number'
              ? question.options[selected]
              : null;
        const correctLabel =
          question.type === 'fill_blank'
            ? question.acceptedAnswers.join(', ')
            : question.type === 'select_all'
              ? normalizeSelectedIndexes(question.answerIndexes)
                  .map((answerIndex) => question.options[answerIndex])
                  .join(', ')
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
            ) : question.type === 'select_all' ? (
              <div className="list">
                {question.options.map((option, optionIndex) => {
                  const selectedIndexes = Array.isArray(selected) ? selected : [];
                  const isChecked = selectedIndexes.includes(optionIndex);
                  const optionIsCorrect = submitted && question.answerIndexes.includes(optionIndex);
                  const optionIsWrong =
                    submitted && isChecked && !question.answerIndexes.includes(optionIndex);

                  return (
                    <label key={optionIndex} className={`option ${optionIsCorrect ? 'correct' : ''} ${optionIsWrong ? 'wrong' : ''}`}>
                      <input
                        type="checkbox"
                        name={`${question.id}-${optionIndex}`}
                        checked={isChecked}
                        onChange={() =>
                          setAnswers((prev) => {
                            const current = Array.isArray(prev[question.id]) ? [...(prev[question.id] as number[])] : [];
                            const next = current.includes(optionIndex)
                              ? current.filter((value) => value !== optionIndex)
                              : [...current, optionIndex];

                            return { ...prev, [question.id]: normalizeSelectedIndexes(next) };
                          })
                        }
                      />
                      <span>{option}</span>
                    </label>
                  );
                })}
              </div>
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
