// components/ThemeToggle/index.tsx
import React from 'react';
import { ToggleWrapper, ToggleCircle } from './styles';

type ThemeToggleProps = {
  isDark: boolean;
  onToggle: () => void;
};

export const ThemeToggle: React.FC<ThemeToggleProps> = ({
  isDark,
  onToggle,
}) => {
  return (
    <ToggleWrapper onClick={onToggle} aria-label='Toggle theme'>
      <ToggleCircle isDark={isDark}>
        <span>{isDark ? '🌙' : '🌞'}</span>
      </ToggleCircle>
    </ToggleWrapper>
  );
};
