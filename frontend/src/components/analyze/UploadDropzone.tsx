import { useCallback, useRef, useState } from 'react';

import { Icon } from '../ui/Icon';

const ACCEPT = 'image/jpeg,image/png,image/webp,image/bmp,image/tiff';
const MAX_BYTES = 10 * 1024 * 1024;

interface Props {
  disabled?: boolean;
  onFile: (file: File) => void;
}

/** Skeuomorphic drop-tray: a recessed slot you drop a photo into. */
export function UploadDropzone({ disabled, onFile }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handle = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      if (!file.type.startsWith('image/')) return setError('Please choose an image file.');
      if (file.size > MAX_BYTES) return setError('That image is larger than 10 MB.');
      setError(null);
      onFile(file);
    },
    [onFile],
  );

  return (
    <div>
      <button
        type="button"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!disabled) handle(e.dataTransfer.files[0]);
        }}
        className={`tray group flex w-full flex-col items-center justify-center gap-3 px-6 py-12 text-center transition-all duration-300 disabled:cursor-not-allowed disabled:opacity-50 ${
          dragging ? 'ring-2 ring-brand ring-offset-2 ring-offset-bg' : ''
        }`}
      >
        <span
          className={`clay-sm flex h-16 w-16 items-center justify-center text-brand transition-transform duration-300 ${
            dragging ? 'scale-110 -rotate-6' : 'group-hover:-translate-y-1'
          }`}
        >
          <Icon name="image" size={28} />
        </span>
        <span className="font-display text-lg font-semibold text-ink">
          Drop an image or click to browse
        </span>
        <span className="text-xs text-ink-faint">
          JPEG · PNG · WebP · BMP · TIFF &nbsp;·&nbsp; up to 10 MB
        </span>
      </button>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onChange={(e) => handle(e.target.files?.[0])}
      />
      {error && (
        <p className="mt-2 flex items-center gap-1.5 text-sm text-poor">
          <Icon name="alert" size={14} /> {error}
        </p>
      )}
    </div>
  );
}
