'use client';

import { useState } from 'react';
import { useFormState, useFormStatus } from 'react-dom';
import { initialActionState } from '@/lib/action-state';
import { deleteCourseAction, deleteQuizAction, updateQuizAction } from '@/lib/actions';

type AdminQuestion = {
  type: 'multiple_choice' | 'select_all' | 'ordering' | 'fill_blank';
  prompt: string;
  options: string[];
  answerIndex: number | null;
  answerIndexes: number[];
  acceptedAnswers: string[];
  explanation: string | null;
  order: number;
};

type AdminQuiz = {
  id: string;
  title: string;
  slug: string;
  description: string | null;
  courseId: string;
  questions: AdminQuestion[];
};

type AdminCourse = {
  id: string;
  name: string;
  code: string;
  slug: string;
  description: string | null;
  quizzes: AdminQuiz[];
};

function quizToJson(quiz: AdminQuiz) {
  return JSON.stringify(
    {
      title: quiz.title,
      slug: quiz.slug,
      description: quiz.description ?? undefined,
      questions: [...quiz.questions]
        .sort((a, b) => a.order - b.order)
        .map((question) =>
          question.type === 'fill_blank'
            ? {
                type: 'fill_blank',
                prompt: question.prompt,
                acceptedAnswers: question.acceptedAnswers,
                explanation: question.explanation ?? undefined
              }
            : question.type === 'ordering'
              ? {
                  type: 'ordering',
                  prompt: question.prompt,
                  options: question.options,
                  answerIndexes: question.answerIndexes,
                  explanation: question.explanation ?? undefined
                }
            : question.type === 'select_all'
              ? {
                  type: 'select_all',
                  prompt: question.prompt,
                  options: question.options,
                  answerIndexes: question.answerIndexes,
                  explanation: question.explanation ?? undefined
                }
            : {
                type: 'multiple_choice',
                prompt: question.prompt,
                options: question.options,
                answerIndex: question.answerIndex ?? 0,
                explanation: question.explanation ?? undefined
              }
        )
    },
    null,
    2
  );
}

function DeleteButton({ label }: { label: string }) {
  const { pending } = useFormStatus();
  return (
    <button className="btn danger" type="submit" disabled={pending}>
      {pending ? 'Working...' : label}
    </button>
  );
}

function SaveButton() {
  const { pending } = useFormStatus();
  return (
    <button className="btn" type="submit" disabled={pending}>
      {pending ? 'Saving...' : 'Save Quiz Changes'}
    </button>
  );
}

function DeleteCourseForm({ courseId }: { courseId: string }) {
  const [state, formAction] = useFormState(deleteCourseAction, initialActionState);

  return (
    <form
      action={formAction}
      className="form"
      onSubmit={(event) => {
        if (!window.confirm('Delete this course and all quizzes inside it? This cannot be undone.')) {
          event.preventDefault();
        }
      }}
    >
      <input type="hidden" name="courseId" value={courseId} />
      <DeleteButton label="Delete Course" />
      {state.message ? <p className={state.ok ? 'success' : 'error'}>{state.message}</p> : null}
    </form>
  );
}

function DeleteQuizForm({ quizId }: { quizId: string }) {
  const [state, formAction] = useFormState(deleteQuizAction, initialActionState);

  return (
    <form
      action={formAction}
      className="form"
      onSubmit={(event) => {
        if (!window.confirm('Delete this quiz? This cannot be undone.')) {
          event.preventDefault();
        }
      }}
    >
      <input type="hidden" name="quizId" value={quizId} />
      <DeleteButton label="Delete Quiz" />
      {state.message ? <p className={state.ok ? 'success' : 'error'}>{state.message}</p> : null}
    </form>
  );
}

function EditQuizForm({
  quiz,
  courses
}: {
  quiz: AdminQuiz;
  courses: Pick<AdminCourse, 'id' | 'code' | 'name'>[];
}) {
  const [state, formAction] = useFormState(updateQuizAction, initialActionState);
  const [jsonText, setJsonText] = useState(() => quizToJson(quiz));

  return (
    <form action={formAction} className="form">
      <input type="hidden" name="quizId" value={quiz.id} />
      <div className="grid two">
        <div>
          <label className="muted">Course</label>
          <select className="select" name="courseId" defaultValue={quiz.courseId} required>
            {courses.map((course) => (
              <option key={course.id} value={course.id}>
                {course.code} - {course.name}
              </option>
            ))}
          </select>
        </div>
        <div className="admin-meta">
          <span className="badge">{quiz.slug}</span>
          <span className="muted">{quiz.questions.length} question{quiz.questions.length === 1 ? '' : 's'}</span>
        </div>
      </div>
      <textarea
        className="textarea admin-json"
        name="jsonText"
        value={jsonText}
        onChange={(event) => setJsonText(event.target.value)}
      />
      <SaveButton />
      {state.message ? <p className={state.ok ? 'success' : 'error'}>{state.message}</p> : null}
    </form>
  );
}

export function AdminCourseManager({ courses }: { courses: AdminCourse[] }) {
  return (
    <section className="card admin-manager">
      <div>
        <h3>Manage content</h3>
        <p className="muted">Edit quiz JSON inline, move a quiz to another course, or delete quizzes and courses.</p>
      </div>

      {courses.length === 0 ? (
        <div className="card">
          <p className="muted">No courses yet. Create a course first to manage quizzes.</p>
        </div>
      ) : (
        <div className="admin-course-list">
          {courses.map((course) => (
            <article key={course.id} className="card admin-course-card">
              <div className="admin-course-head">
                <div>
                  <span className="badge">{course.code}</span>
                  <h4>{course.name}</h4>
                  <p className="muted">{course.description || 'No course description.'}</p>
                </div>
                <DeleteCourseForm courseId={course.id} />
              </div>

              {course.quizzes.length === 0 ? (
                <p className="muted">No quizzes uploaded for this course yet.</p>
              ) : (
                <div className="admin-quiz-list">
                  {course.quizzes.map((quiz) => (
                    <details key={quiz.id} className="admin-quiz-card">
                      <summary>
                        <div>
                          <strong>{quiz.title}</strong>
                          <p className="muted">{quiz.description || 'No quiz description.'}</p>
                        </div>
                        <span className="badge">Edit Quiz</span>
                      </summary>
                      <div className="admin-quiz-actions">
                        <EditQuizForm quiz={quiz} courses={courses} />
                        <DeleteQuizForm quizId={quiz.id} />
                      </div>
                    </details>
                  ))}
                </div>
              )}
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
