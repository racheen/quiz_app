import React from 'react';
import { ButtonContainer, BackToHomeButton } from './style'

type ReturnButtonProps = {
    label: string;
    onClick: () => void;
  };


export const ReturnButton: React.FC<ReturnButtonProps> = ({
    label,
    onClick,
}) => {
    return (
      <ButtonContainer>
        <BackToHomeButton onClick={onClick}>
          {label}
        </BackToHomeButton>
      </ButtonContainer>
    );
  };
  