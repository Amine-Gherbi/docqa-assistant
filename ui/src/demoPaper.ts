/** The real paper and the three recorded, extractive example answers. */

import rawPaper from './fixtures/real-paper.json';
import type { AskResponse } from './types';

export interface DemoQuestion {
  question: string;
  answer: AskResponse;
}

export interface RealPaper {
  doc_id: string;
  title: string;
  pdf: string;
  source: string;
  page_count: number;
  generated_with: string;
  questions: DemoQuestion[];
}

export const DEMO_PAPER = rawPaper as unknown as RealPaper;
export const DEMO_QUESTIONS: readonly DemoQuestion[] = DEMO_PAPER.questions;
export const DEFAULT_DEMO_QUESTION = DEMO_QUESTIONS[0] as DemoQuestion;

/** Return the recorded answer for a question, ignoring case and outer space. */
export function demoForQuestion(question: string): DemoQuestion | null {
  const needle = question.trim().toLowerCase();
  return (
    DEMO_QUESTIONS.find(
      (example) => example.question.trim().toLowerCase() === needle,
    ) ?? null
  );
}
