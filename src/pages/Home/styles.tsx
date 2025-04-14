import styled from 'styled-components';

export const Container = styled.div`
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  margin: 40px;
  justify-content: center;

  @media (max-width: 768px) {
    flex-direction: column;
    padding: 20px 30px;
    margin: 0;
  }
`;

export const ProgressText = styled.p`
  font-size: 1rem;
  color: #333;
  text-align: center;
  margin-top: 20px;
  font-weight: bold;
`;

export const ProgressWrapper = styled.div`
  display: flex;
  flex-direction: column; /* Align children vertically */
  align-items: center; /* Center children horizontally */
  justify-content: center; /* Center children vertically */
  gap: 20px; /* Add some space between the question and the progress text */
  width: 100%; /* Ensure it takes full width */
  padding: 20px;

  @media (max-width: 768px) {
    padding: 0px;
  }
`;
export const IndexButton = styled.button<{
  answered: boolean;
  current: boolean;
  isCorrect: boolean;
}>`
  padding: 0.5rem 0.75rem;
  margin: 0.25rem;
  border-radius: 0.25rem;
  border: 2px solid ${({ theme }) => theme.colors.accent};
  cursor: pointer;
  font-weight: ${({ current }) => (current ? 'bold' : 'normal')};
  color: ${({ current }) => (current ? 'white' : 'black')};
  background-color: ${({ answered, current, isCorrect, theme }) => {
    if (current) return theme.colors.primary;
    if (!answered) return theme.colors.lightGray;
    return isCorrect ? theme.colors.accent : theme.colors.lightRed;
  }};

  &:hover {
    opacity: 0.8;
  }
`;

export const MenuContainer = styled.div`
  margin-top: 1rem;
`;

export const IndexButtonContainer = styled.div`
  margin-top: 1rem;
  display: flex;
  flex-wrap: wrap;

  @media (max-width: 768px) {
    flex-wrap: nowrap;
    overflow-x: auto;
    padding: 0;
    gap: 0.5rem;
    width: 300px;

    /* Optional: enhance mobile scroll UX */
    scrollbar-width: thin;
    scrollbar-color: #ccc transparent;

    &::-webkit-scrollbar {
      height: 6px;
    }

    &::-webkit-scrollbar-thumb {
      background: #ccc;
      border-radius: 3px;
    }

    &::-webkit-scrollbar-track {
      background: transparent;
    }
  }
`;

export const SubmitButton = styled.button`
  padding: 0.5rem 0.75rem;
  margin: 0.25rem;
  border-radius: 0.25rem;
  border: none;
  background-color: ${({ theme }) => theme.colors.lightGray};
  cursor: pointer;
  border: 2px solid ${({ theme }) => theme.colors.accent};

  &:hover {
    opacity: 0.8;
  }
`;
