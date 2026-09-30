// Grey placeholders shown while an admin page loads.

export function MetricsSkeleton({ label }: { label: string }) {
  return (
    <section className="admin-metrics" aria-label={label} aria-busy="true">
      <div className="admin-skeleton metric" />
      <div className="admin-skeleton metric" />
      <div className="admin-skeleton metric" />
      <div className="admin-skeleton metric" />
    </section>
  );
}

// Fills a panel body (table, list or form) while its data loads.
export function BlockSkeleton({ label }: { label: string }) {
  return <div className="admin-skeleton block" aria-label={label} aria-busy="true" />;
}
