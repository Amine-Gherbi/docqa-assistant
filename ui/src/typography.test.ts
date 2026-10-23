import { describe, expect, it } from 'vitest';

import { formatQuoted, nestQuotes } from './typography';

describe('quotation typography', () => {
  it('turns quote marks nested inside a quotation into single quotes', () => {
    expect(nestQuotes('the rare term “quokka”')).toBe('the rare term ‘quokka’');
    expect(nestQuotes('attention "Scaled Dot-Product Attention"')).toBe(
      'attention ‘Scaled Dot-Product Attention’',
    );
  });

  it('wraps a quotation with outer double quotes and inner single quotes', () => {
    expect(formatQuoted('the rare term “quokka”')).toBe(
      '“the rare term ‘quokka’”',
    );
  });
});
