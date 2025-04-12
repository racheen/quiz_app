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
`;
