// components/ThemeToggle/styles.tsx
import styled from 'styled-components';

export const ToggleWrapper = styled.button`
  background: ${({ theme }) => theme.colors.lightGray};
  border: 2px solid ${({ theme }) => theme.colors.lightGreen};
  border-radius: 30px;
  cursor: pointer;
  display: flex;
  align-items: center;
  padding: 4px;
  width: 60px;
  height: 30px;
  transition: background 0.3s ease;
  position: fixed;
  bottom: 20px;
  right: 20px;
  z-index: 1000;
`;

export const ToggleCircle = styled.div<{ isDark: boolean }>`
  background: ${({ theme }) => theme.colors.primary};
  height: 22px;
  width: 22px;
  border-radius: 50%;
  position: absolute;
  left: ${({ isDark }) => (isDark ? '4px' : 'calc(100% - 26px)')};
  transition: left 0.3s ease;
  display: flex;
  justify-content: center;
  align-items: center;

  span {
    font-size: 16px;
    color: ${({ theme }) => theme.colors.primary};
  }
`;
