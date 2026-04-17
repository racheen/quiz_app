import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function main() {
  const course = await prisma.course.upsert({
    where: { code: 'CST2216' },
    update: {},
    create: {
      name: 'Machine Learning 2',
      code: 'CST2216',
      slug: 'cst2216',
      description: 'Sample seeded course'
    }
  });

  const existing = await prisma.quiz.findFirst({
    where: { courseId: course.id, slug: 'sample-mock-1' }
  });

  if (!existing) {
    await prisma.quiz.create({
      data: {
        title: 'Sample Mock 1',
        slug: 'sample-mock-1',
        description: 'Seeded starter quiz',
        courseId: course.id,
        questions: {
          create: [
            {
              order: 1,
              type: 'multiple_choice',
              prompt: 'What does NLP stand for?',
              options: ['Natural Language Processing', 'Numeric Language Parsing', 'Neural Logic Protocol', 'Natural Logic Processing'],
              answerIndex: 0,
              explanation: 'NLP stands for Natural Language Processing.'
            },
            {
              order: 2,
              type: 'multiple_choice',
              prompt: 'Which model type is easier to interpret?',
              options: ['Black-box', 'White-box', 'Ensemble-only', 'Transformer-only'],
              answerIndex: 1,
              explanation: 'White-box models are easier to interpret.'
            },
            {
              order: 3,
              type: 'fill_blank',
              prompt: 'Fill in the blank: The data used to train a model is called the ______ set.',
              acceptedAnswers: ['training', 'training set'],
              explanation: 'Training data is used to fit the model.'
            }
          ]
        }
      }
    });
  }
}

main()
  .then(async () => {
    await prisma.$disconnect();
  })
  .catch(async (error) => {
    console.error(error);
    await prisma.$disconnect();
    process.exit(1);
  });
