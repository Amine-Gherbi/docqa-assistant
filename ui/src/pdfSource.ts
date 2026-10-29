/**
 * Options for every PDF.js load in this app.
 *
 * `isEvalSupported: false` stops PDF.js from evaluating embedded JavaScript.
 * `withCredentials: true` sends the signed client cookie with the owner-scoped
 * `/documents/{id}/file` request; the API refuses any other client.
 */
export function pdfSourceOptions(url: string): {
  url: string;
  verbosity: number;
  isEvalSupported: false;
  withCredentials: true;
} {
  return {
    url,
    verbosity: 0,
    isEvalSupported: false,
    withCredentials: true,
  };
}
