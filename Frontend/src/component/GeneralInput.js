import React from "react";
import {Input, Tooltip  } from "antd";
import '../css/Layout.css';
function NumberFormat(value) {
    value += '';
    const list = value.split('.');
    const prefix = list[0].charAt(0) === '-' ? '-' : '';
    let number = prefix ? list[0].slice(1) : list[0];
    let result = '';
    while (number.length > 3) {
      result = `,${number.slice(-3)}${result}`;
      number = number.slice(0, number.length - 3);
    }
    if (number) {
      result = number + result;
    }
    return `${prefix}${result}${list[1] ? `.${list[1]}` : ''}`;
  }
  class GeneralInput extends React.Component {
    onChange = e => {
      const { value } = e.target;
      const reg = /^-?[0-9]*(\.[0-9]*)?$/;
      if ((!isNaN(value) && reg.test(value)) || value === '' || value === '-') {
        this.props.onChange(value);
      }
    };
  
    onBlur = () => {
      const { value, onBlur, onChange } = this.props;
      let valueTemp = value;
      if (value.charAt(value.length - 1) === '.' || value === '-') {
        valueTemp = value.slice(0, -1);
      }
      onChange(valueTemp.replace(/0*(\d+)/, '$1'));
      if (onBlur) {
        onBlur();
      }
    };
  
    render() {
      const { value } = this.props;
      const title = value ? (
        <span className="numeric-input-title">{value !== '-' ? NumberFormat(value) : '-'}</span>
      ) : (
        'Input a valid number please!'
      );
      return (
        <Tooltip
          trigger={['focus']}
          title={title}
          placement="topLeft"
          overlayClassName="numeric-input"
        >
          <Input
            {...this.props}
            onChange={this.onChange}
            onBlur={this.onBlur}
            placeholder="Input a value..."
            maxLength={25}
          />
        </Tooltip>
      );
    }
  }
  export default GeneralInput