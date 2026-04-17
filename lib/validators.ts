import { z } from 'zod';

export const createCourseSchema = z.object({
  name: z.string().min(2),
  code: z.string().min(2),
  description: z.string().optional()
});

const multipleChoiceQuestionSchema = z.object({
  type: z.literal('multiple_choice').optional().default('multiple_choice'),
  prompt: z.string().min(5),
  options: z.array(z.string().min(1)).min(2),
  answerIndex: z.number().int().min(0),
  explanation: z.string().optional()
});

const fillBlankQuestionSchema = z.object({
  type: z.literal('fill_blank'),
  prompt: z.string().min(5),
  acceptedAnswers: z.array(z.string().min(1)).min(1),
  explanation: z.string().optional()
});

export const quizQuestionSchema = z.discriminatedUnion('type', [
  multipleChoiceQuestionSchema,
  fillBlankQuestionSchema
]);

export const quizUploadSchema = z.object({
  title: z.string().min(2),
  slug: z.string().optional(),
  description: z.string().optional(),
  questions: z.array(quizQuestionSchema).min(1)
});

export type QuizUploadInput = z.infer<typeof quizUploadSchema>;
