import { DatabaseUnavailable } from '@/components/database-unavailable';
import { isDatabaseUnavailableError } from '@/lib/db-errors';
import Link from 'next/link';
import { prisma } from '@/lib/prisma';

export const dynamic = 'force-dynamic';

export default async function CoursesPage() {
  try {
    const courses = await prisma.course.findMany({
      orderBy: { createdAt: 'desc' },
      include: { quizzes: { orderBy: { createdAt: 'desc' } } }
    });

    return (
      <main className="container grid" style={{ gap: 20 }}>
        <div className="card">
          <h1>Courses</h1>
          <p className="muted">Each course can contain multiple quizzes or mock exams.</p>
        </div>
        {courses.map((course) => (
          <div className="card" key={course.id}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
              <div>
                <span className="badge">{course.code}</span>
                <h2>{course.name}</h2>
                <p className="muted">{course.description || 'No description yet.'}</p>
              </div>
              <Link className="btn secondary" href={`/courses/${course.slug}`}>Open course</Link>
            </div>
            <div className="list" style={{ marginTop: 16 }}>
              {course.quizzes.length === 0 ? (
                <p className="muted">No quizzes uploaded yet.</p>
              ) : course.quizzes.map((quiz) => (
                <Link key={quiz.id} href={`/courses/${course.slug}/${quiz.slug}`} className="option">
                  <div>
                    <strong>{quiz.title}</strong>
                    <div className="muted">{quiz.description || 'No description'}</div>
                  </div>
                </Link>
              ))}
            </div>
          </div>
        ))}
      </main>
    );
  } catch (error) {
    if (isDatabaseUnavailableError(error)) {
      return <DatabaseUnavailable title="Courses unavailable" />;
    }

    throw error;
  }
}
