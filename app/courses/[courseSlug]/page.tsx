import { DatabaseUnavailable } from '@/components/database-unavailable';
import { isDatabaseUnavailableError } from '@/lib/db-errors';
import { notFound } from 'next/navigation';
import Link from 'next/link';
import { prisma } from '@/lib/prisma';

export const dynamic = 'force-dynamic';

export default async function CourseDetailPage({ params }: { params: { courseSlug: string } }) {
  try {
    const course = await prisma.course.findUnique({
      where: { slug: params.courseSlug },
      include: {
        quizzes: {
          orderBy: { createdAt: 'desc' },
          include: { _count: { select: { questions: true } } }
        }
      }
    });

    if (!course) notFound();

    return (
      <main className="container grid" style={{ gap: 20 }}>
        <div className="card">
          <span className="badge">{course.code}</span>
          <h1>{course.name}</h1>
          <p className="muted">{course.description || 'No description yet.'}</p>
        </div>

        <div className="grid two">
          {course.quizzes.length === 0 ? (
            <div className="card"><p className="muted">No quizzes uploaded yet.</p></div>
          ) : course.quizzes.map((quiz) => (
            <Link key={quiz.id} href={`/courses/${course.slug}/${quiz.slug}`} className="card">
              <h3>{quiz.title}</h3>
              <p className="muted">{quiz.description || 'No description.'}</p>
              <p>{quiz._count.questions} questions</p>
            </Link>
          ))}
        </div>
      </main>
    );
  } catch (error) {
    if (isDatabaseUnavailableError(error)) {
      return <DatabaseUnavailable title="Course unavailable" />;
    }

    throw error;
  }
}
