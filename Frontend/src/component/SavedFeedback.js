import React from "react"
import {Table, Divider, Button, Input, Modal, Collapse, Card, Col, Row,notification} from 'antd';
import axios from "axios";
import Highlighter from 'react-highlight-words';
import PropTypes from 'prop-types';
import { SearchOutlined } from '@ant-design/icons';
import '../css/Layout.css';
import { TweenOneGroup } from 'rc-tween-one';
const TableContext = React.createContext(false);
const { Panel } = Collapse;

const SavedFeedback = () => {

  return (
    <h1>Saved Feedback</h1>
  )

}

export default SavedFeedback