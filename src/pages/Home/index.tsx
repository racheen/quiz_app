import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { MainTopic, questions } from '../../data/questions';
import QuestionCard from '../../components/QuestionCard';
import QuizResult from '../../components/QuizResult';
import TopicSelector from '../../components/TopicSelector';
import {
  Container,
  IndexButton,
  IndexButtonContainer,
  MenuContainer,
  ProgressText,
  ProgressWrapper,
  SubmitButton,
} from './styles';
import { ReturnButton } from '../../components/ReturnButton';
import { Question } from '../../types/question';
import Modal from '../../components/Modal';
import { topicHierarchy } from '../../data/TopicMap';

export default function HomePage() {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [isFinished, setIsFinished] = useState(false);
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [answers, setAnswers] = useState<(string | null)[]>([]);
  const [incorrectAnswers, setIncorrectAnswers] = useState<
    {
      question: string;
      chosenAnswer: string;
      correctAnswer: string;
      explanation: string;
    }[]
  >([]);
  const [shuffledQuestions, setShuffledQuestions] = useState<Question[]>([]);
  const [showModal, setShowModal] = useState(false);
  const [modalMessage, setModalMessage] = useState('');
  const [selectedMainTopic, setSelectedMainTopic] = useState<MainTopic | null>(
    null
  );

  const navigate = useNavigate();

  const mainTopics = Object.values(MainTopic);

  const confirmSubmit = () => {
    setShowModal(false);
    setIsFinished(true);
  };

  const cancelSubmit = () => {
    setShowModal(false);
  };

  const handleSubmit = () => {
    const unansweredCount = answers.filter((ans) => ans === null).length;

    if (unansweredCount > 0) {
      setModalMessage(
        `You still have ${unansweredCount} unanswered question${
          unansweredCount > 1 ? 's' : ''
        }. Are you sure you want to submit?`
      );
    } else {
      setModalMessage(
        'You are done with the quiz, do you want to view summary your answers?'
      );
    }

    setShowModal(true);
  };

  function shuffleArray(array: Question[]) {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  }

  const handleAnswer = (answer: string) => {
    const currentQuestion = shuffledQuestions[currentIndex];

    // Store answer
    const updatedAnswers = [...answers];
    updatedAnswers[currentIndex] = answer;
    setAnswers(updatedAnswers);

    if (answer === currentQuestion.answer) {
      setScore(score + 1);
    } else {
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

    if (!updatedAnswers.includes(null)) {
      setModalMessage('Are you sure you want to submit your answers?');
      setShowModal(true);
    }
  };

  const handleNext = () => {
    const next = currentIndex + 1;
    if (next < shuffledQuestions.length) {
      setCurrentIndex(next);
    }
  };

  const handleSelectTopic = (topic: string) => {
    const topicQuestions = questions.filter((q) =>
      Array.isArray(q.topic) ? q.topic.includes(topic) : q.topic === topic
    );
    const shuffled = shuffleArray(topicQuestions);
    setShuffledQuestions(shuffled);
    setSelectedTopic(topic);
    setCurrentIndex(0);
    setScore(0);
    setIsFinished(false);
    setAnswers(Array(shuffled.length).fill(null));
    setIncorrectAnswers([]);

    setSelectedTopic(topic);
    setCurrentIndex(0);
    setScore(0);
    setIsFinished(false);
    setAnswers(Array(shuffled.length).fill(null));
    setIncorrectAnswers([]);
  };

  const handleBackToHome = () => {
    setSelectedMainTopic(null);
    setSelectedTopic(null);
    if (window.location.pathname === '/') {
      window.location.reload();
    } else {
      navigate('/');
    }
  };

  const isMainTopic = (value: string): boolean => {
    return Object.values(MainTopic).includes(value as unknown as MainTopic);
  };

  return (
    <Container>
      {!selectedMainTopic ? (
        <TopicSelector
          topics={mainTopics}
          onSelectTopic={(main) => {
            if (isMainTopic(main)) {
              setSelectedMainTopic(main as unknown as MainTopic);
            } else {
              console.warn(`Invalid main topic selected: ${main}`);
            }
          }}
        />
      ) : !selectedTopic ? (
        <>
          <ReturnButton
            onClick={() => setSelectedMainTopic(null)}
            label='← Back to Main Topics'
          />
          <TopicSelector
            topics={topicHierarchy[selectedMainTopic]}
            onSelectTopic={handleSelectTopic}
          />
        </>
      ) : !isFinished && shuffledQuestions.length !== 0 ? (
        <ProgressWrapper>
          <ProgressText>
            Question {currentIndex + 1} of {shuffledQuestions.length}
          </ProgressText>
          <QuestionCard
            question={shuffledQuestions[currentIndex]}
            onAnswer={handleAnswer}
            selectedAnswer={answers[currentIndex]} // Optional: highlight selected
          />
          {answers[currentIndex] !== null && (
            <SubmitButton key='next' onClick={handleNext}>
              Next
            </SubmitButton>
          )}
          <MenuContainer>
            <IndexButtonContainer>
              {shuffledQuestions.map((_, idx) => (
                <IndexButton
                  key={idx}
                  onClick={() => setCurrentIndex(idx)}
                  answered={answers[idx] !== null}
                  current={idx === currentIndex}
                  isCorrect={
                    answers[idx] !== null
                      ? answers[idx] === shuffledQuestions[idx].answer
                      : false
                  }
                >
                  {idx + 1}
                </IndexButton>
              ))}
              <SubmitButton onClick={handleSubmit}>Submit Quiz</SubmitButton>
            </IndexButtonContainer>
          </MenuContainer>
          <ReturnButton onClick={handleBackToHome} label={'Return to Topics'} />
          {showModal && (
            <Modal
              message={modalMessage}
              onConfirm={confirmSubmit}
              onCancel={cancelSubmit}
            />
          )}
        </ProgressWrapper>
      ) : (
        <>
          <QuizResult
            score={score}
            total={shuffledQuestions.length}
            incorrectAnswers={incorrectAnswers}
          />
          <ReturnButton onClick={handleBackToHome} label={'Back to Home'} />
        </>
      )}
    </Container>
  );
}
