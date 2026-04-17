import { DatabaseUnavailable } from '@/components/database-unavailable';
import { isDatabaseUnavailableError } from '@/lib/db-errors';
import Link from 'next/link';
import { prisma } from '@/lib/prisma';

export const dynamic = 'force-dynamic';

export default async function HomePage() {
  try {
    const courses = await prisma.course.findMany({
      orderBy: { createdAt: 'desc' },
      include: {
        _count: {
          select: { quizzes: true }
        }
      }
    });

    return (
      <main className="container grid" style={{ gap: 24 }}>
        <section className="hero card">
          <span className="badge">niqi is subelior</span>
          <h1>Quiz App for mock exams</h1>
          <p>
            Create a course, upload a quiz JSON file, and instantly publish multiple tests under that course.
            This version is built for local Postgres with Docker and is easy to deploy later.
          </p>
          <div style={{ display: 'flex', gap: 12, marginTop: 18, flexWrap: 'wrap' }}>
            <Link className="btn" href="/admin">Open Admin</Link>
            <Link className="btn secondary" href="/courses">Browse Courses</Link>
          </div>
        </section>

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
      return <DatabaseUnavailable />;
    }

    throw error;
  }
}
