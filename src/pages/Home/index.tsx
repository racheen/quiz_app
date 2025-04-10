import React, {useState} from 'react';
import { questions } from '../../data/questions';
import QuestionCard from '../../components/QuestionCard';
import QuizResult from '../../components/QuizResult';
import { Container } from './styles';

export default function HomePage() {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [isFinished, setIsFinished] = useState(false);

  const handleAnswer = (answer: string) => {
    if (answer === questions[currentIndex].answer) {
      setScore(score + 1);
    }
    const next = currentIndex + 1;
    if (next < questions.length) {
      setCurrentIndex(next);
    } else {
      setIsFinished(true);
    }
  };

  return (
    <Container>
      {!isFinished ? (
        <QuestionCard
          question={questions[currentIndex]}
          onAnswer={handleAnswer}
        />
      ) : (
        <QuizResult score={score} total={questions.length} />
      )}
    </Container>
  );
}
