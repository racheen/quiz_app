import { DatabaseUnavailable } from '@/components/database-unavailable';
import { isDatabaseUnavailableError } from '@/lib/db-errors';
import Link from 'next/link';
import { prisma } from '@/lib/prisma';

export const dynamic = 'force-dynamic';

export default async function CoursesPage() {
  try {
    const courses = await prisma.course.findMany({
      orderBy: { createdAt: 'desc' },
      include: {
        quizzes: {
          orderBy: { createdAt: 'desc' },
          take: 3
        },
        _count: {
          select: { quizzes: true }
        }
      }
    });

    return (
      <main className="container grid" style={{ gap: 20 }}>
        <div className="card">
          <h1>Courses</h1>
          <p className="muted">Each course can contain multiple quizzes or mock exams.</p>
        </div>
        <section className="grid two">
          {courses.length === 0 ? (
            <div className="card">
              <h3>No courses yet</h3>
              <p className="muted">Go to the admin page to create your first course and upload a quiz.</p>
            </div>
          ) : (
            courses.map((course) => (
              <Link key={course.id} href={`/courses/${course.slug}`} className="card">
                <span className="badge">{course.code}</span>
                <h3>{course.name}</h3>
                <p className="muted">{course.description || 'No description yet.'}</p>
                <p><strong>{course._count.quizzes}</strong> quiz{course._count.quizzes === 1 ? '' : 'zes'}</p>
              </Link>
            ))
          )}
        </section>
      </main>
    );
  } catch (error) {
    if (isDatabaseUnavailableError(error)) {
      return <DatabaseUnavailable title="Courses unavailable" />;
    }

    throw error;
  }
}
