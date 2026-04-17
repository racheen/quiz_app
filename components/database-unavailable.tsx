type DatabaseUnavailableProps = {
  title?: string;
  message?: string;
};

const defaultMessage =
  'Postgres is not reachable at localhost:5434. Start the Docker database and run `npm run db:push`, then refresh.';

export function DatabaseUnavailable({
  title = 'Database unavailable',
  message = defaultMessage
}: DatabaseUnavailableProps) {
  return (
    <main className="container">
      <section className="card" style={{ maxWidth: 720 }}>
        <h1>{title}</h1>
        <p className="muted">{message}</p>
      </section>
    </main>
  );
}
