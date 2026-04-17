'use client';

import Link from 'next/link';
import { useEffect, useMemo, useRef, useState } from 'react';

type Question = {
  id: string;
  type: 'multiple_choice' | 'select_all' | 'ordering' | 'fill_blank';
  prompt: string;
  options: string[];
  answerIndex: number | null;
  answerIndexes: number[];
  acceptedAnswers: string[];
  explanation: string | null;
  order: number;
};

type CourseQuiz = {
  title: string;
  href: string;
  isCurrent: boolean;
};

type AnswerValue = string | number | number[];

type QuestionStatus = {
  question: Question;
  index: number;
  selected: AnswerValue | undefined;
  wasAnswered: boolean;
  isCorrect: boolean;
  selectedLabel: string | null;
  correctLabel: string;
};

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

function orderingMatches(selected: number[], expected: number[]) {
  return selected.length === expected.length && selected.every((value, index) => value === expected[index]);
}

function getOrderingLabel(options: string[], order: number[]) {
  return order.map((optionIndex, index) => `${index + 1}. ${options[optionIndex]}`).join(' | ');
}

function wasQuestionAnswered(question: Question, selected: AnswerValue | undefined) {
  return question.type === 'fill_blank'
    ? typeof selected === 'string' && selected.trim().length > 0
    : question.type === 'select_all' || question.type === 'ordering'
      ? Array.isArray(selected) && selected.length > 0
      : typeof selected === 'number';
}

function isQuestionCorrect(question: Question, selected: AnswerValue | undefined) {
  if (question.type === 'fill_blank') {
    return (
      typeof selected === 'string' &&
      question.acceptedAnswers.some(
        (acceptedAnswer) => normalizeFillBlankAnswer(acceptedAnswer) === normalizeFillBlankAnswer(selected)
      )
    );
  }

  if (question.type === 'select_all') {
    return Array.isArray(selected) && answersMatch(selected, question.answerIndexes);
  }

  if (question.type === 'ordering') {
    return Array.isArray(selected) && orderingMatches(selected, question.answerIndexes);
  }

  return typeof selected === 'number' && selected === question.answerIndex;
}

function getSelectedLabel(question: Question, selected: AnswerValue | undefined) {
  if (question.type === 'fill_blank') {
    return typeof selected === 'string' ? selected : null;
  }

  if (question.type === 'select_all') {
    return Array.isArray(selected) && selected.length > 0
      ? normalizeSelectedIndexes(selected).map((answerIndex) => question.options[answerIndex]).join(', ')
      : null;
  }

  if (question.type === 'ordering') {
    return Array.isArray(selected) && selected.length > 0 ? getOrderingLabel(question.options, selected) : null;
  }

  return typeof selected === 'number' ? question.options[selected] : null;
}

function getCorrectLabel(question: Question) {
  if (question.type === 'fill_blank') {
    return question.acceptedAnswers.join(', ');
  }

  if (question.type === 'select_all') {
    return normalizeSelectedIndexes(question.answerIndexes)
      .map((answerIndex) => question.options[answerIndex])
      .join(', ');
  }

  if (question.type === 'ordering') {
    return getOrderingLabel(question.options, question.answerIndexes);
  }

  return question.answerIndex !== null ? question.options[question.answerIndex] : '';
}

export function QuizRunner({
  courseTitle,
  quizTitle,
  questions,
  quizzesHref,
  courseQuizzes,
  nextQuizHref,
  nextQuizTitle
}: {
  courseTitle: string;
  quizTitle: string;
  questions: Question[];
  quizzesHref: string;
  courseQuizzes: CourseQuiz[];
  nextQuizHref: string | null;
  nextQuizTitle: string | null;
}) {
  const [answers, setAnswers] = useState<Record<string, AnswerValue>>({});
  const [submitted, setSubmitted] = useState(false);
  const [activeQuestionId, setActiveQuestionId] = useState<string | null>(questions[0]?.id ?? null);
  const summaryRef = useRef<HTMLDivElement | null>(null);
  const questionRefs = useRef<Record<string, HTMLDivElement | null>>({});

  const questionStatuses = useMemo(
    () =>
      questions.map((question, index) => {
        const selected = answers[question.id];

        return {
          question,
          index,
          selected,
          wasAnswered: wasQuestionAnswered(question, selected),
          isCorrect: isQuestionCorrect(question, selected),
          selectedLabel: getSelectedLabel(question, selected),
          correctLabel: getCorrectLabel(question)
        };
      }),
    [answers, questions]
  );

  const score = useMemo(
    () => questionStatuses.reduce((total, status) => total + (status.isCorrect ? 1 : 0), 0),
    [questionStatuses]
  );

  const answeredCount = questionStatuses.filter((status) => status.wasAnswered).length;
  const unansweredCount = questions.length - answeredCount;
  const percentage = questions.length === 0 ? 0 : Math.round((score / questions.length) * 100);

  useEffect(() => {
    setActiveQuestionId(questions[0]?.id ?? null);
  }, [questions]);

  useEffect(() => {
    if (submitted) {
      summaryRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [submitted]);

  useEffect(() => {
    const questionNodes = questions
      .map((question) => questionRefs.current[question.id])
      .filter((node): node is HTMLDivElement => Boolean(node));

    if (questionNodes.length === 0 || typeof IntersectionObserver === 'undefined') {
      return undefined;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const visibleIds = entries
          .filter((entry) => entry.isIntersecting)
          .map((entry) => entry.target.getAttribute('data-question-id'))
          .filter((value): value is string => Boolean(value));

        if (visibleIds.length === 0) {
          return;
        }

        const nextActiveQuestion = questions.find((question) => visibleIds.includes(question.id));
        if (nextActiveQuestion) {
          setActiveQuestionId(nextActiveQuestion.id);
        }
      },
      {
        rootMargin: '-18% 0px -60% 0px',
        threshold: [0.15, 0.4, 0.7]
      }
    );

    questionNodes.forEach((node) => observer.observe(node));

    return () => observer.disconnect();
  }, [questions]);

  const goToQuestion = (questionId: string) => {
    setActiveQuestionId(questionId);
    questionRefs.current[questionId]?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const scrollToSummary = () => {
    summaryRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <div className="quiz-shell">
      <aside className="card quiz-sidebar">
        <div className="quiz-sidebar-stack">
          <div className="quiz-sidebar-section">
            <p className="quiz-sidebar-kicker">Course</p>
            <Link href={quizzesHref} className="quiz-course-home">
              <strong>{courseTitle}</strong>
              <span className="muted">Back to this course&apos;s quiz list</span>
            </Link>
          </div>

          <div className="quiz-sidebar-section">
            <p className="quiz-sidebar-kicker">Quizzes</p>
            <div className="quiz-sidebar-panel">
              <div className="quiz-course-list" id="quiz-course-navigation">
                {courseQuizzes.map((courseQuiz, index) => (
                  <Link
                    key={courseQuiz.href}
                    href={courseQuiz.href}
                    className={`quiz-course-link ${courseQuiz.isCurrent ? 'active' : ''}`}
                    aria-current={courseQuiz.isCurrent ? 'page' : undefined}
                  >
                    <span className="quiz-course-index">Quiz {index + 1}</span>
                    <span className="quiz-course-title">{courseQuiz.title}</span>
                    {courseQuiz.isCurrent ? <span className="quiz-course-state">Current quiz</span> : null}
                  </Link>
                ))}
              </div>
            </div>
          </div>

          <div className="quiz-sidebar-section">
            <div className="quiz-sidebar-section-head">
              <p className="quiz-sidebar-kicker">Questions</p>
              <span className="muted">{answeredCount} answered</span>
            </div>
            <div className="quiz-sidebar-panel">
              <div className="quiz-question-jump" id="quiz-question-navigation">
                {questionStatuses.map((status) => (
                  <button
                    type="button"
                    key={status.question.id}
                    className={`quiz-question-link ${activeQuestionId === status.question.id ? 'active' : ''} ${status.wasAnswered ? 'answered' : 'unanswered'}`}
                    onClick={() => goToQuestion(status.question.id)}
                    aria-current={activeQuestionId === status.question.id ? 'true' : undefined}
                  >
                    {`Question ${status.index + 1}`}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </aside>

      <div className="quiz-main">
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

          {questionStatuses.map((status) => {
            const { question, index, selected, wasAnswered, isCorrect, selectedLabel, correctLabel } = status;

            return (
              <div
                className="card quiz-question-card"
                key={question.id}
                data-question-id={question.id}
                ref={(node) => {
                  questionRefs.current[question.id] = node;
                }}
              >
                <h3>{index + 1}. {question.prompt}</h3>
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
                      const optionIsWrong = submitted && isChecked && !question.answerIndexes.includes(optionIndex);

                      return (
                        <label
                          key={optionIndex}
                          className={`option ${optionIsCorrect ? 'correct' : ''} ${optionIsWrong ? 'wrong' : ''}`}
                        >
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
                ) : question.type === 'ordering' ? (
                  <div className="list">
                    {(Array.isArray(selected) ? selected : question.options.map((_, optionIndex) => optionIndex)).map(
                      (optionIndex, orderIndex, currentOrder) => {
                        const moveOption = (direction: -1 | 1) => {
                          setAnswers((prev) => {
                            const baseOrder = Array.isArray(prev[question.id])
                              ? [...(prev[question.id] as number[])]
                              : question.options.map((_, indexValue) => indexValue);
                            const targetIndex = orderIndex + direction;

                            if (targetIndex < 0 || targetIndex >= baseOrder.length) {
                              return prev;
                            }

                            [baseOrder[orderIndex], baseOrder[targetIndex]] = [baseOrder[targetIndex], baseOrder[orderIndex]];
                            return { ...prev, [question.id]: baseOrder };
                          });
                        };

                        const positionIsCorrect = submitted && question.answerIndexes[orderIndex] === optionIndex;

                        return (
                          <div
                            key={optionIndex}
                            className={`ordering-option ${submitted ? (positionIsCorrect ? 'correct' : 'wrong') : ''}`}
                          >
                            <div className="ordering-option-copy">
                              <span className="badge">#{orderIndex + 1}</span>
                              <span>{question.options[optionIndex]}</span>
                            </div>
                            <div className="ordering-controls">
                              <button
                                type="button"
                                className="btn secondary"
                                onClick={() => moveOption(-1)}
                                disabled={submitted || orderIndex === 0}
                              >
                                Up
                              </button>
                              <button
                                type="button"
                                className="btn secondary"
                                onClick={() => moveOption(1)}
                                disabled={submitted || orderIndex === currentOrder.length - 1}
                              >
                                Down
                              </button>
                            </div>
                          </div>
                        );
                      }
                    )}
                  </div>
                ) : (
                  <div className="list">
                    {question.options.map((option, optionIndex) => {
                      const optionIsCorrect = submitted && optionIndex === question.answerIndex;
                      const optionIsWrong = submitted && selected === optionIndex && optionIndex !== question.answerIndex;

                      return (
                        <label
                          key={optionIndex}
                          className={`option ${optionIsCorrect ? 'correct' : ''} ${optionIsWrong ? 'wrong' : ''}`}
                        >
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
                      {isCorrect ? 'Correct.' : wasAnswered ? 'Not quite.' : 'No answer selected.'}
                    </p>
                    {wasAnswered ? (
                      <p className="muted">
                        <strong>Your answer:</strong> {selectedLabel}
                      </p>
                    ) : null}
                    <p className="muted">
                      <strong>Correct answer:</strong> {correctLabel}
                    </p>
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

          <div className="card quiz-footer">
            <div className="quiz-footer-actions">
              <Link href={quizzesHref} className="btn secondary">
                Back to quizzes
              </Link>
              {submitted ? (
                <button type="button" className="btn secondary" onClick={scrollToSummary}>
                  Review results
                </button>
              ) : (
                <button type="button" className="btn" onClick={() => setSubmitted(true)}>
                  Submit quiz
                </button>
              )}
              {submitted && nextQuizHref ? (
                <Link href={nextQuizHref} className="btn">
                  Next quiz{nextQuizTitle ? `: ${nextQuizTitle}` : ''}
                </Link>
              ) : null}
            </div>
            <p className="muted">
              {submitted
                ? 'Use the sidebar to review other quizzes in the course or jump straight back to any question.'
                : 'Use the sidebar to switch quizzes in this course or jump directly to any question while you work.'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
