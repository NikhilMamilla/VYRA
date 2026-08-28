/** VYRA wordmark: a claymorphic aperture mark + the name in the display face. */
export function Wordmark({ compact = false }: { compact?: boolean }) {
  return (
    <span className="flex items-center gap-2.5">
      <span className="clay-sm grid h-8 w-8 place-items-center rounded-xl">
        <svg viewBox="0 0 24 24" className="h-4 w-4">
          <defs>
            <linearGradient id="vyra-mark" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stopColor="rgb(var(--c-brand))" />
              <stop offset="1" stopColor="rgb(var(--c-brand-2))" />
            </linearGradient>
          </defs>
          <g fill="none" stroke="url(#vyra-mark)" strokeWidth="2" strokeLinecap="round">
            <circle cx="12" cy="12" r="8.5" />
            <path d="M12 3.5 12 12 5.8 17.2 M12 12 19.2 15.8" />
          </g>
        </svg>
      </span>
      {!compact && (
        <span className="font-display text-lg font-bold tracking-tight text-ink">VYRA</span>
      )}
    </span>
  );
}
