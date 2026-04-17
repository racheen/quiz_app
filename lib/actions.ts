'use server';

import { Prisma } from '@prisma/client';
import { revalidatePath } from 'next/cache';
import type { ActionState } from '@/lib/action-state';
import { prisma } from '@/lib/prisma';
import { isDatabaseUnavailableError } from '@/lib/db-errors';
import { slugify } from '@/lib/utils';
import { createCourseSchema, quizUploadSchema } from '@/lib/validators';

function getCourseMutationErrorMessage(error: unknown) {
  if (isDatabaseUnavailableError(error)) {
    return 'The database is currently unavailable. Check your DATABASE_URL and database connection, then try again.';
  }

  if (error instanceof Prisma.PrismaClientKnownRequestError && error.code === 'P2002') {
    return 'That course code is already in use.';
  }

  return 'Please check the form and try again.';
}

function getQuizMutationErrorMessage(error: unknown) {
  if (isDatabaseUnavailableError(error)) {
    return 'The database is currently unavailable. Check your DATABASE_URL and database connection, then try again.';
  }

  if (error instanceof Prisma.PrismaClientKnownRequestError && error.code === 'P2002') {
    return 'That quiz slug is already in use for this course.';
  }

  return 'Make sure the JSON matches the required format and try again.';
}

function parseQuizPayload(jsonText: string) {
  const raw = JSON.parse(jsonText);
  const parsed = quizUploadSchema.parse(raw);
  const slug = slugify(parsed.slug || parsed.title);

  const invalidAnswer = parsed.questions.find(
    (question) =>
      question.type === 'multiple_choice' &&
      (question.answerIndex < 0 || question.answerIndex >= question.options.length)
  );

  if (invalidAnswer) {
    throw new Error('INVALID_ANSWER_INDEX');
  }

  const invalidSelectAllAnswer = parsed.questions.find(
    (question) =>
      question.type === 'select_all' &&
      (question.answerIndexes.some(
        (answerIndex) => answerIndex < 0 || answerIndex >= question.options.length
      ) ||
        new Set(question.answerIndexes).size !== question.answerIndexes.length)
  );

  if (invalidSelectAllAnswer) {
    throw new Error('INVALID_ANSWER_INDEXES');
  }

  return { parsed, slug };
}

export async function createCourseAction(_: ActionState, formData: FormData): Promise<ActionState> {
  try {
    const parsed = createCourseSchema.parse({
      name: formData.get('name'),
      code: formData.get('code'),
      description: formData.get('description') || undefined
    });

    const slug = slugify(parsed.code);

    await prisma.course.create({
      data: {
        name: parsed.name,
        code: parsed.code.toUpperCase(),
        slug,
        description: parsed.description
      }
    });

    revalidatePath('/');
    revalidatePath('/courses');
    revalidatePath('/admin');

    return { ok: true, message: 'Course created.' };
  } catch (error) {
    console.error(error);
    return { ok: false, message: `Could not create course. ${getCourseMutationErrorMessage(error)}` };
  }
}

export async function uploadQuizAction(_: ActionState, formData: FormData): Promise<ActionState> {
  try {
    const courseId = String(formData.get('courseId') || '');
    const jsonText = String(formData.get('jsonText') || '');

    if (!courseId || !jsonText) {
      return { ok: false, message: 'Course and quiz JSON are required.' };
    }

    let parsed;
    let slug;
    try {
      const result = parseQuizPayload(jsonText);
      parsed = result.parsed;
      slug = result.slug;
    } catch (error) {
      if (error instanceof Error && error.message === 'INVALID_ANSWER_INDEX') {
        return { ok: false, message: 'One or more questions has an invalid answerIndex.' };
      }

      if (error instanceof Error && error.message === 'INVALID_ANSWER_INDEXES') {
        return { ok: false, message: 'One or more select-all questions has invalid answerIndexes.' };
      }

      throw error;
    }

    await prisma.quiz.create({
      data: {
        courseId,
        title: parsed.title,
        slug,
        description: parsed.description,
        questions: {
          create: parsed.questions.map((question, index) => ({
            order: index + 1,
            type: question.type,
            prompt: question.prompt,
            options:
              question.type === 'multiple_choice' || question.type === 'select_all'
                ? question.options
                : undefined,
            answerIndex: question.type === 'multiple_choice' ? question.answerIndex : undefined,
            answerIndexes: question.type === 'select_all' ? question.answerIndexes : [],
            acceptedAnswers: question.type === 'fill_blank' ? question.acceptedAnswers : undefined,
            explanation: question.explanation
          }))
        }
      }
    });

    revalidatePath('/');
    revalidatePath('/courses');
    revalidatePath('/admin');

    return { ok: true, message: 'Quiz uploaded successfully.' };
  } catch (error) {
    console.error(error);
    return {
      ok: false,
      message: `Could not upload quiz. ${getQuizMutationErrorMessage(error)}`
    };
  }
}

export async function updateQuizAction(_: ActionState, formData: FormData): Promise<ActionState> {
  try {
    const quizId = String(formData.get('quizId') || '');
    const courseId = String(formData.get('courseId') || '');
    const jsonText = String(formData.get('jsonText') || '');

    if (!quizId || !courseId || !jsonText) {
      return { ok: false, message: 'Quiz details and quiz JSON are required.' };
    }

    let parsed;
    let slug;
    try {
      const result = parseQuizPayload(jsonText);
      parsed = result.parsed;
      slug = result.slug;
    } catch (error) {
      if (error instanceof Error && error.message === 'INVALID_ANSWER_INDEX') {
        return { ok: false, message: 'One or more questions has an invalid answerIndex.' };
      }

      if (error instanceof Error && error.message === 'INVALID_ANSWER_INDEXES') {
        return { ok: false, message: 'One or more select-all questions has invalid answerIndexes.' };
      }

      throw error;
    }

    await prisma.quiz.update({
      where: { id: quizId },
      data: {
        courseId,
        title: parsed.title,
        slug,
        description: parsed.description,
        questions: {
          deleteMany: {},
          create: parsed.questions.map((question, index) => ({
            order: index + 1,
            type: question.type,
            prompt: question.prompt,
            options:
              question.type === 'multiple_choice' || question.type === 'select_all'
                ? question.options
                : undefined,
            answerIndex: question.type === 'multiple_choice' ? question.answerIndex : undefined,
            answerIndexes: question.type === 'select_all' ? question.answerIndexes : [],
            acceptedAnswers: question.type === 'fill_blank' ? question.acceptedAnswers : undefined,
            explanation: question.explanation
          }))
        }
      }
    });

    revalidatePath('/');
    revalidatePath('/courses');
    revalidatePath('/admin');

    return { ok: true, message: 'Quiz updated successfully.' };
  } catch (error) {
    console.error(error);
    return {
      ok: false,
      message: `Could not update quiz. ${getQuizMutationErrorMessage(error)}`
    };
  }
}

export async function deleteQuizAction(_: ActionState, formData: FormData): Promise<ActionState> {
  try {
    const quizId = String(formData.get('quizId') || '');

    if (!quizId) {
      return { ok: false, message: 'Quiz ID is required.' };
    }

    await prisma.quiz.delete({
      where: { id: quizId }
    });

    revalidatePath('/');
    revalidatePath('/courses');
    revalidatePath('/admin');

    return { ok: true, message: 'Quiz deleted.' };
  } catch (error) {
    console.error(error);
    return {
      ok: false,
      message: `Could not delete quiz. ${getQuizMutationErrorMessage(error)}`
    };
  }
}

export async function deleteCourseAction(_: ActionState, formData: FormData): Promise<ActionState> {
  try {
    const courseId = String(formData.get('courseId') || '');

    if (!courseId) {
      return { ok: false, message: 'Course ID is required.' };
    }

    await prisma.course.delete({
      where: { id: courseId }
    });

    revalidatePath('/');
    revalidatePath('/courses');
    revalidatePath('/admin');

    return { ok: true, message: 'Course deleted.' };
  } catch (error) {
    console.error(error);
    return {
      ok: false,
      message: `Could not delete course. ${getCourseMutationErrorMessage(error)}`
    };
  }
}
