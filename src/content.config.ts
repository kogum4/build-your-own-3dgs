import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const chapters = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/chapters' }),
  schema: z.object({
    title: z.string(),
    chapter: z.number().int().min(1).max(16),
    lang: z.enum(['ja', 'en']),
    description: z.string(),
    pyodide: z.boolean().default(false),
  }),
});

export const collections = { chapters };
