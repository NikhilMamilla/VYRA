import { useCallback, useRef, useState } from 'react';

const ACCEPT = 'image/jpeg,image/png,image/webp,image/bmp,image/tiff';
const MAX_BYTES = 10 * 1024 * 1024;

interface Props {
  disabled?: boolean;
  onFile: (file: File) => void;
}

/** Drag-and-drop / click file picker with client-side pre-checks. */
export function UploadDropzone({ disabled, onFile }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const handle = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      if (!file.type.startsWith('image/')) {
        setLocalError('Please choose an image file.');
        return;
      }
      if (file.size > MAX_BYTES) {
        setLocalError('That image is larger than 10 MB.');
        return;
      }
      setLocalError(null);
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
        className={`flex w-full flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-6 py-10 text-center transition ${
          dragging
            ? 'border-slate-900 bg-slate-50'
            : 'border-slate-300 hover:border-slate-400 hover:bg-slate-50'
        } disabled:cursor-not-allowed disabled:opacity-50`}
      >
        <svg className="h-8 w-8 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5"
          />
        </svg>
        <span className="font-medium text-slate-700">Drop an image or click to browse</span>
        <span className="text-xs text-slate-400">JPEG, PNG, WebP, BMP or TIFF · up to 10 MB</span>
      </button>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onChange={(e) => handle(e.target.files?.[0])}
      />
      {localError && <p className="mt-2 text-sm text-red-600">{localError}</p>}
    </div>
  );
}
