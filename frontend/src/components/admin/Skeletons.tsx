// Grey placeholders shown while an admin page loads.

export function OverviewSkeleton({ label }: { label: string }) {
  return (
    <section className="admin-metrics-strip" aria-label={label}>
      <div className="admin-skeleton metric" />
      <div className="admin-skeleton metric" />
      <div className="admin-skeleton metric" />
      <div className="admin-skeleton metric" />
    </section>
  );
}

export function TableSkeleton({ label }: { label: string }) {
  return <div className="admin-skeleton table" aria-label={label} />;
}

export function DocumentSkeleton({ label }: { label: string }) {
  return <div className="admin-skeleton document" aria-label={label} />;
}

export function ConfigurationSkeleton({ label }: { label: string }) {
  return (
    <section className="admin-configuration-layout" aria-label={label}>
      <div className="admin-skeleton form" />
      <div className="admin-skeleton form" />
      <div className="admin-skeleton form" />
    </section>
  );
}
