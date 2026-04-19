import { DatabaseUnavailable } from '@/components/database-unavailable';
import { isDatabaseUnavailableError } from '@/lib/db-errors';
import { notFound } from 'next/navigation';
import Link from 'next/link';
import { prisma } from '@/lib/prisma';

export const dynamic = 'force-dynamic';

const QUIZZES_PER_PAGE = 6;

function parsePageParam(pageParam: string | string[] | undefined) {
  const rawValue = Array.isArray(pageParam) ? pageParam[0] : pageParam;
  const page = Number(rawValue);

  if (!Number.isFinite(page) || page < 1) {
    return 1;
  }

  return Math.floor(page);
}

function buildCoursePageHref(courseSlug: string, page: number) {
  return page === 1 ? `/courses/${courseSlug}` : `/courses/${courseSlug}?page=${page}`;
}

function getVisiblePages(currentPage: number, totalPages: number) {
  const pages = new Set([1, totalPages, currentPage - 1, currentPage, currentPage + 1]);

  return [...pages].filter((page) => page >= 1 && page <= totalPages).sort((a, b) => a - b);
}

export default async function CourseDetailPage({
  params,
  searchParams
}: {
  params: { courseSlug: string };
  searchParams?: { page?: string | string[] };
}) {
  try {
    const course = await prisma.course.findUnique({
      where: { slug: params.courseSlug },
      select: {
        id: true,
        slug: true,
        code: true,
        name: true,
        description: true,
        _count: {
          select: { quizzes: true }
        }
      }
    });

    if (!course) notFound();

    const requestedPage = parsePageParam(searchParams?.page);
    const totalPages = Math.max(1, Math.ceil(course._count.quizzes / QUIZZES_PER_PAGE));
    const currentPage = Math.min(requestedPage, totalPages);
    const visiblePages = getVisiblePages(currentPage, totalPages);
    const quizzes = await prisma.quiz.findMany({
      where: { courseId: course.id },
      orderBy: { createdAt: 'desc' },
      include: { _count: { select: { questions: true } } },
      skip: (currentPage - 1) * QUIZZES_PER_PAGE,
      take: QUIZZES_PER_PAGE
    });
    const firstQuizIndex = course._count.quizzes === 0 ? 0 : (currentPage - 1) * QUIZZES_PER_PAGE + 1;
    const lastQuizIndex = course._count.quizzes === 0 ? 0 : firstQuizIndex + quizzes.length - 1;

    return (
      <main className="container grid" style={{ gap: 20 }}>
        <div className="card">
          <span className="badge">{course.code}</span>
          <h1>{course.name}</h1>
          <p className="muted">{course.description || 'No description yet.'}</p>
          <p className="muted">
            <strong>{course._count.quizzes}</strong> quiz{course._count.quizzes === 1 ? '' : 'zes'}
            {course._count.quizzes > 0 ? ` • Page ${currentPage} of ${totalPages}` : ''}
          </p>
        </div>

        <div className="grid two">
          {quizzes.length === 0 ? (
            <div className="card"><p className="muted">No quizzes uploaded yet.</p></div>
          ) : quizzes.map((quiz) => (
            <Link key={quiz.id} href={`/courses/${course.slug}/${quiz.slug}`} className="card">
              <h3>{quiz.title}</h3>
              <p className="muted">{quiz.description || 'No description.'}</p>
              <p>{quiz._count.questions} questions</p>
            </Link>
          ))}
        </div>

        {course._count.quizzes > QUIZZES_PER_PAGE ? (
          <nav className="pagination" aria-label="Quiz pagination">
            <p className="pagination-summary">
              Showing {firstQuizIndex}-{lastQuizIndex} of {course._count.quizzes} quizzes
            </p>
            <div className="pagination-links">
              {currentPage > 1 ? (
                <Link className="pagination-link" href={buildCoursePageHref(course.slug, currentPage - 1)}>
                  Previous
                </Link>
              ) : null}

              {visiblePages.map((page, index) => {
                const previousPage = visiblePages[index - 1];
                const showEllipsis = previousPage && page - previousPage > 1;

                return (
                  <div key={page} className="pagination-item">
                    {showEllipsis ? <span className="pagination-ellipsis">...</span> : null}
                    <Link
                      aria-current={page === currentPage ? 'page' : undefined}
                      className={`pagination-link ${page === currentPage ? 'active' : ''}`}
                      href={buildCoursePageHref(course.slug, page)}
                    >
                      {page}
                    </Link>
                  </div>
                );
              })}

              {currentPage < totalPages ? (
                <Link className="pagination-link" href={buildCoursePageHref(course.slug, currentPage + 1)}>
                  Next
                </Link>
              ) : null}
            </div>
          </nav>
        ) : null}
      </main>
    );
  } catch (error) {
    if (isDatabaseUnavailableError(error)) {
      return <DatabaseUnavailable title="Course unavailable" />;
    }

    throw error;
  }
}
