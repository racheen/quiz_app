import { Button } from 'antd';
import styled from 'styled-components';

export const StyledButton = styled(Button)`
  padding: 9px 16px;
  height: auto;
  box-sizing: border-box;
  border: 3px solid ${(props) => props.theme.colors.lightGreen};
  border-radius: 4px;
  border-color: ${(props) => props.theme.colors.lightGreen};
  background: ${(props) => props.theme.colors.white};
  font-size: ${(props) => props.theme.fontSizes.large};
  font-weight: ${(props) => props.theme.fontWeights.regular};

  &.ant-btn.ant-btn-primary {
    color: ${(props) => props.theme.colors.lightGreen};
  }

  &.ant-btn-primary:hover {
    filter: drop-shadow(0px 4px 4px rgba(0, 0, 0, 0.25));
    cursor: pointer;
  }

  &.ant-btn-primary:focus {
    box-shadow: inset 0px 4px 4px rgba(0, 0, 0, 0.25);
  }

  &.ant-btn-lg,
  &.ant-btn {
    padding: 12px 24px;
    line-height: 1;
  }

  &.ant-btn-lg {
    padding: 16px 24px;
  }

  &.ant-btn-sm {
    padding: 8px 10px;
  }

  margin: 10px;
`;
