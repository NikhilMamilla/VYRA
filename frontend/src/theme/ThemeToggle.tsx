import { Icon } from '../components/ui/Icon';
import { useTheme } from './ThemeProvider';

/** Neumorphic pill toggle: the knob sits in an inset track. */
export function ThemeToggle() {
  const { theme, toggle } = useTheme();
  const isDark = theme === 'dark';

  return (
    <button
      type="button"
      onClick={toggle}
      role="switch"
      aria-checked={isDark}
      aria-label={`Switch to ${isDark ? 'light' : 'dark'} theme`}
      className="neu-inset relative flex h-9 w-[4.25rem] items-center rounded-full px-1"
    >
      <span
        className={`neu-sm flex h-7 w-7 items-center justify-center rounded-full bg-surface-solid text-ink transition-transform duration-300 ${
          isDark ? 'translate-x-[2.25rem]' : 'translate-x-0'
        }`}
      >
        <Icon name={isDark ? 'moon' : 'sun'} size={15} />
      </span>
      <Icon
        name="sun"
        size={13}
        className={`absolute left-2 text-ink-faint transition-opacity ${isDark ? 'opacity-100' : 'opacity-0'}`}
      />
      <Icon
        name="moon"
        size={13}
        className={`absolute right-2 text-ink-faint transition-opacity ${isDark ? 'opacity-0' : 'opacity-100'}`}
      />
    </button>
  );
}
