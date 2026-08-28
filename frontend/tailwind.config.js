/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: ['class', '[data-theme="dark"]'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Inter Variable"', 'Inter', 'system-ui', 'sans-serif'],
        display: ['"Sora Variable"', 'Sora', '"Inter Variable"', 'system-ui', 'sans-serif'],
      },
      colors: {
        bg: 'rgb(var(--c-bg) / <alpha-value>)',
        'bg-alt': 'rgb(var(--c-bg-alt) / <alpha-value>)',
        surface: 'rgb(var(--c-surface) / <alpha-value>)',
        'surface-solid': 'rgb(var(--c-surface-solid) / <alpha-value>)',
        clay: 'rgb(var(--c-clay) / <alpha-value>)',
        ink: 'rgb(var(--c-ink) / <alpha-value>)',
        'ink-soft': 'rgb(var(--c-ink-soft) / <alpha-value>)',
        'ink-faint': 'rgb(var(--c-ink-faint) / <alpha-value>)',
        line: 'rgb(var(--c-line) / <alpha-value>)',
        brand: 'rgb(var(--c-brand) / <alpha-value>)',
        'brand-2': 'rgb(var(--c-brand-2) / <alpha-value>)',
        good: 'rgb(var(--c-good) / <alpha-value>)',
        ok: 'rgb(var(--c-ok) / <alpha-value>)',
        degraded: 'rgb(var(--c-degraded) / <alpha-value>)',
        poor: 'rgb(var(--c-poor) / <alpha-value>)',
        'tier-real': 'rgb(var(--c-tier-real) / <alpha-value>)',
        'tier-synth': 'rgb(var(--c-tier-synth) / <alpha-value>)',
        'tier-screen': 'rgb(var(--c-tier-screen) / <alpha-value>)',
      },
      borderRadius: {
        clay: '1.75rem',
        'clay-sm': '1.25rem',
      },
      boxShadow: {
        glass: 'var(--sh-glass)',
        clay: 'var(--sh-clay)',
        'clay-sm': 'var(--sh-clay-sm)',
        neu: 'var(--sh-neu)',
        'neu-inset': 'var(--sh-neu-inset)',
        'neu-sm': 'var(--sh-neu-sm)',
        glow: 'var(--sh-glow)',
      },
      backgroundImage: {
        brand: 'linear-gradient(120deg, rgb(var(--c-brand)), rgb(var(--c-brand-2)))',
        'brand-soft':
          'linear-gradient(120deg, rgb(var(--c-brand) / 0.14), rgb(var(--c-brand-2) / 0.14))',
      },
      keyframes: {
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '0%': { transform: 'translateX(-120%)' },
          '100%': { transform: 'translateX(420%)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-6px)' },
        },
        'sweep-in': {
          '0%': { strokeDashoffset: 'var(--dash)' },
        },
      },
      animation: {
        'fade-up': 'fade-up 0.5s cubic-bezier(0.22, 1, 0.36, 1) both',
        float: 'float 6s ease-in-out infinite',
      },
    },
  },
  plugins: [],
};
