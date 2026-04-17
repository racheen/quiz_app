import './globals.css';
import Link from 'next/link';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Quiz App',
  description: 'Next.js quiz app with Postgres and Docker'
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <nav className="nav">
          <Link href="/"><strong>Quiz App</strong></Link>
          <div className="nav-links">
            <Link href="/courses">Courses</Link>
            <Link href="/admin">Admin</Link>
            <Link href="/admin/format">JSON Format</Link>
          </div>
        </nav>
        {children}
      </body>
    </html>
  );
}
