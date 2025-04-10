import styled from 'styled-components';

export const Container = styled.div`
  display: flex;
  flex-direction: row;
  margin: 40px;
  justifycontent: 'space-between';

  @media (max-width: 768px) {
    flex-direction: column;
    padding: 20px 30px;
    margin: 0;
  }
`;
