import '@testing-library/jest-dom/vitest';

// jsdom has no createObjectURL / revokeObjectURL.
if (!URL.createObjectURL) {
  URL.createObjectURL = () => 'blob:mock';
  URL.revokeObjectURL = () => {};
}
