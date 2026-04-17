'use client';

import { useFormState, useFormStatus } from 'react-dom';
import { initialActionState } from '@/lib/action-state';
import { createCourseAction } from '@/lib/actions';

function SubmitButton() {
  const { pending } = useFormStatus();
  return (
    <button className="btn" type="submit" disabled={pending}>
      {pending ? 'Creating...' : 'Add Course'}
    </button>
  );
}

export function CourseForm() {
  const [state, formAction] = useFormState(createCourseAction, initialActionState);

  return (
    <form action={formAction} className="form card">
      <div>
        <h3>Add course</h3>
        <p className="muted">Create a course first, then upload quizzes into it.</p>
      </div>
      <input className="input" name="name" placeholder="Course name" required />
      <input className="input" name="code" placeholder="Course code (e.g. CST2216)" required />
      <textarea className="textarea" name="description" placeholder="Optional description" />
      <SubmitButton />
      {state.message ? <p className={state.ok ? 'success' : 'error'}>{state.message}</p> : null}
    </form>
  );
}
