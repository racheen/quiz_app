import { AdminCourseManager } from '@/components/admin-course-manager';
import { CourseForm } from '@/components/course-form';
import { DatabaseUnavailable } from '@/components/database-unavailable';
import { isDatabaseUnavailableError } from '@/lib/db-errors';
import { QuizUploadForm } from '@/components/quiz-upload-form';
import { prisma } from '@/lib/prisma';

export const dynamic = 'force-dynamic';

export default async function AdminPage() {
  try {
    const courses = await prisma.course.findMany({
      orderBy: { createdAt: 'desc' },
      select: {
        id: true,
        name: true,
        code: true,
        slug: true,
        description: true,
        quizzes: {
          orderBy: { createdAt: 'desc' },
          select: {
            id: true,
            title: true,
            slug: true,
            description: true,
            courseId: true,
            questions: {
              orderBy: { order: 'asc' },
              select: {
                type: true,
                prompt: true,
                options: true,
                answerIndex: true,
                answerIndexes: true,
                acceptedAnswers: true,
                explanation: true,
                order: true
              }
            }
          }
        }
      }
    });

    return (
      <main className="container grid" style={{ gap: 20 }}>
        <section className="grid two">
          <CourseForm />
          <QuizUploadForm courses={courses.map((course) => ({ id: course.id, name: course.name, code: course.code }))} />
        </section>
        <AdminCourseManager
          courses={courses.map((course) => ({
            ...course,
            quizzes: course.quizzes.map((quiz) => ({
              ...quiz,
              questions: quiz.questions.map((question) => ({
                ...question,
                type: question.type as 'multiple_choice' | 'select_all' | 'fill_blank',
                options: (question.options as string[] | null) ?? [],
                acceptedAnswers: (question.acceptedAnswers as string[] | null) ?? [],
                answerIndex: question.answerIndex,
                answerIndexes: question.answerIndexes
              }))
            }))
          }))}
        />
      </main>
    );
  } catch (error) {
    if (isDatabaseUnavailableError(error)) {
      return (
        <DatabaseUnavailable
          title="Admin unavailable"
          message="The database is currently unavailable, so course creation and quiz upload are disabled. Check your `DATABASE_URL`, make sure your hosted Postgres instance is running, apply the Prisma schema, then refresh."
        />
      );
    }

    throw error;
  }
}
