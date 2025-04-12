import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { questions } from '../../data/questions';
import QuestionCard from '../../components/QuestionCard';
import QuizResult from '../../components/QuizResult';
import TopicSelector from '../../components/TopicSelector';
import { Container, ProgressText, ProgressWrapper } from './styles';
import { ReturnButton } from '../../components/ReturnButton';
import { Question } from '../../types/question';

export default function HomePage() {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [isFinished, setIsFinished] = useState(false);
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);

  const navigate = useNavigate(); // Get the navigate function

  // Track incorrect answers
  const [incorrectAnswers, setIncorrectAnswers] = useState<
    {
      question: string;
      chosenAnswer: string;
      correctAnswer: string;
      explanation: string;
    }[]
  >([]);

  // Get unique topics from the questions
  const topics = Array.from(new Set(questions.map((q) => q.topic)));

  // Filter questions based on the selected topic
  const filteredQuestions = selectedTopic
    ? shuffleArray(questions.filter((q) => q.topic === selectedTopic))
    : [];

  // Fisher-Yates shuffle function
  function shuffleArray(array: Question[]) {
    const shuffled = [...array]; // Create a copy to avoid mutating the original
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]; // Swap elements
    }
    return shuffled;
  }

  const handleAnswer = (answer: string) => {
    const currentQuestion = filteredQuestions[currentIndex];
    if (answer === currentQuestion.answer) {
      setScore(score + 1);
    } else {
      // If the answer is incorrect, store the wrong answer with explanation
      setIncorrectAnswers([
        ...incorrectAnswers,
        {
          question: currentQuestion.question,
          chosenAnswer: answer,
          correctAnswer: currentQuestion.answer,
          explanation: currentQuestion.explanation,
        },
      ]);
    }

    const next = currentIndex + 1;
    if (next < filteredQuestions.length) {
      setCurrentIndex(next);
    } else {
      setIsFinished(true);
    }
  };

  const handleBackToHome = () => {
    if (window.location.pathname === '/') {
      window.location.reload(); // Refresh the page
    } else {
      navigate('/'); // Navigate to home if not already on it
    }
  };

  const handleSelectTopic = (topic: string) => {
    setSelectedTopic(topic);
    setCurrentIndex(0); // Reset to the first question of the selected topic
    setScore(0);
    setIsFinished(false);
    setIncorrectAnswers([]); // Reset incorrect answers when selecting a new topic
  };

  return (
    <Container>
      {!selectedTopic ? (
        <TopicSelector topics={topics} onSelectTopic={handleSelectTopic} />
      ) : !isFinished ? (
        <ProgressWrapper>
          <ProgressText>
            Question {currentIndex + 1} of {filteredQuestions.length}
          </ProgressText>
          <QuestionCard
            question={filteredQuestions[currentIndex]}
            onAnswer={handleAnswer}
          />
          <ReturnButton onClick={handleBackToHome} label={'Return to Topics'} />
        </ProgressWrapper>
      ) : (
        <>
          <QuizResult
            score={score}
            total={filteredQuestions.length}
            incorrectAnswers={incorrectAnswers}
          />{' '}
          <ReturnButton onClick={handleBackToHome} label={'Back to Home'} />
        </>
      )}
    </Container>
  );
}
