import styled, { css } from 'styled-components';

type CardProps = {
  isProject?: boolean;
};

export const Container = styled.div<CardProps>`
  display: flex;
  width: 300px;
  background: ${(props) => props.theme.colors.lightGray};
  box-shadow: ${(props) => props.theme.shadows.dropShadowCard};
  flex-direction: column;
  margin: 20px 5px;

  ${(props) =>
    props.isProject &&
    css`
      width: 400px;
    `}

  a {
    text-decoration: none;
    display: flex;
    flex-direction: column;

    :hover {
      filter: drop-shadow(0px 4px 4px rgba(0, 0, 0, 0.25));
    }
  }
`;

export const CardContentLayout = styled.div`
  padding: 43px 12px 12px 30px;
  color: ${(props) => props.theme.colors.darkGreen};
  font-weight: ${(props) => props.theme.fontWeights.regular};
`;

export const Title = styled(CardContentLayout)<CardProps>`
  font-weight: ${(props) => props.theme.fontWeights.semiBold};
  font-size: ${(props) => props.theme.fontSizes.large};
  height: 50px;

  ${(props) =>
    props.isProject &&
    css`
      height: 10px;
    `}
`;

export const MediumGrayText = styled.span`
  color: ${(props) => props.theme.colors.mediumGray};
`;

export const LightGreenText = styled.span`
  color: ${(props) => props.theme.colors.lightGreen};
`;

export const Content = styled(CardContentLayout)`
  padding: 10px 12px 12px 30px;
  font-size: ${(props) => props.theme.fontSizes.medium};
  color: ${(props) => props.theme.colors.mediumGray};
  height: 100px;
`;

export const Subcontent = styled(CardContentLayout)`
  font-size: ${(props) => props.theme.fontSizes.small};
  padding-bottom: 30px;
`;
