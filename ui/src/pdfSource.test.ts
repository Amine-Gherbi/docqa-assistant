import { describe, expect, it } from 'vitest';

import { pdfSourceOptions } from './pdfSource';

describe('pdfSourceOptions', () => {
  it('disables PDF JavaScript evaluation and sends the session cookie', () => {
    expect(pdfSourceOptions('/paper.pdf')).toEqual({
      url: '/paper.pdf',
      verbosity: 0,
      isEvalSupported: false,
      withCredentials: true,
    });
  });
});
