import styled from 'styled-components';

export const Overlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  z-index: 1000;
`;

export const ModalContainer = styled.div`
  background: white;
  padding: 2rem;
  max-width: 500px;
  margin: 10% auto;
  border-radius: 0.5rem;
  box-shadow: 0 0 10px #00000050;
`;
