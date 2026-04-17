import { DatabaseUnavailable } from '@/components/database-unavailable';
import { isDatabaseUnavailableError } from '@/lib/db-errors';
import { notFound } from 'next/navigation';
import { prisma } from '@/lib/prisma';
import { QuizRunner } from '@/components/quiz-runner';

export const dynamic = 'force-dynamic';

export default async function QuizPage({ params }: { params: { courseSlug: string; quizSlug: string } }) {
  try {
    const course = await prisma.course.findUnique({
      where: { slug: params.courseSlug },
      include: {
        quizzes: {
          orderBy: { createdAt: 'desc' },
          select: {
            slug: true,
            title: true
          }
        }
      }
    });
    if (!course) notFound();

    const quiz = await prisma.quiz.findUnique({
      where: {
        courseId_slug: {
          courseId: course.id,
          slug: params.quizSlug
        }
      },
      include: {
        questions: {
          orderBy: { order: 'asc' }
        }
      }
    });

    if (!quiz) notFound();

    const currentQuizIndex = course.quizzes.findIndex((courseQuiz) => courseQuiz.slug === quiz.slug);
    const nextQuiz = currentQuizIndex >= 0 ? course.quizzes[currentQuizIndex + 1] : null;

    const questions = quiz.questions.map((question) => ({
      id: question.id,
      type: question.type as 'multiple_choice' | 'select_all' | 'ordering' | 'fill_blank',
      prompt: question.prompt,
      options: (question.options as string[] | null) ?? [],
      answerIndex: question.answerIndex,
      answerIndexes: question.answerIndexes,
      acceptedAnswers: (question.acceptedAnswers as string[] | null) ?? [],
      explanation: question.explanation,
      order: question.order
    }));

    return (
      <main className="container">
        <QuizRunner
          courseTitle={course.name}
          quizTitle={quiz.title}
          questions={questions}
          quizzesHref={`/courses/${course.slug}`}
          courseQuizzes={course.quizzes.map((courseQuiz) => ({
            title: courseQuiz.title,
            href: `/courses/${course.slug}/${courseQuiz.slug}`,
            isCurrent: courseQuiz.slug === quiz.slug
          }))}
          nextQuizHref={nextQuiz ? `/courses/${course.slug}/${nextQuiz.slug}` : null}
          nextQuizTitle={nextQuiz?.title ?? null}
        />
      </main>
    );
  } catch (error) {
    if (isDatabaseUnavailableError(error)) {
      return <DatabaseUnavailable title="Quiz unavailable" />;
    }

    throw error;
  }
}
