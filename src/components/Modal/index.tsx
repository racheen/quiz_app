import React from 'react';
import { ModalContainer, Overlay } from './style';

type ModalProps = {
  message: string;
  onConfirm: () => void;
  onCancel: () => void;
};

export default function Modal({ message, onConfirm, onCancel }: ModalProps) {
  return (
    <Overlay>
      <ModalContainer>
        <p>{message}</p>
        <div
          style={{
            marginTop: '1rem',
            display: 'flex',
            justifyContent: 'flex-end',
            gap: '1rem',
          }}
        >
          <button onClick={onCancel}>Cancel</button>
          <button onClick={onConfirm}>Confirm</button>
        </div>
      </ModalContainer>
    </Overlay>
  );
}
