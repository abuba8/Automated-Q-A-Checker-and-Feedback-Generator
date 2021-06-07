import React, { useState } from 'react';
import { Form, Input, Icon, Button, Select, Avatar } from 'antd';
import { connect } from 'react-redux';
import { NavLink, useHistory } from 'react-router-dom';
import axios from "axios";
import Notifications, {notify} from 'react-notify-toast';
const {Option} = Select;
const FormItem = Form.Item;

const RegistrationForm = () => {
  
  const history = useHistory();
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [errorMSg, setErrMsg] = useState('')


  const handleSubmit = (e) => {
    e.preventDefault();
    axios.post('http://127.0.0.1:8000/register', {
      username,
      email,
      password,
    },{
      headers: {
        "Content-Type": "application/json",
      }
    })
    .then(res => {
      notify.show("Signup Successfull")
      history.push('/login')
      console.log('res', res)
    }).catch (e => {
      setErrMsg('Something went wrong')
      console.log('Error', e)
    })
  }
  
  return (
    <Form onSubmit={handleSubmit}>
      <Notifications/>
      <FormItem>   
        <Input 
          prefix={<Icon type="user"  />} 
          placeholder="Username" 
          style={{ color: 'rgba(0,0,0,.25)' }}  
          value = { username }
          onChange = { e => setUsername( e.target.value )}
        />
      </FormItem>
      
      <FormItem>
        <Input 
          prefix={<Icon type="mail" />} 
          placeholder="Email" 
          style={{ color: 'rgba(0,0,0,.25)' }}
          value= { email }
          onChange = { e => setEmail( e.target.value )}
        />
      </FormItem>

      <FormItem>
        <Input 
          prefix={<Icon type="lock" />} 
          type="password" 
          placeholder="Password" 
          style={{ color: 'rgba(0,0,0,.25)' }}
          value= { password }
          onChange = { e => setPassword(e.target.value)}
        />
      </FormItem>

      <FormItem>
        <Input prefix={<Icon type="lock" />} 
          type="password" 
          placeholder="Password" 
          style={{ color: 'rgba(0,0,0,.25)' }}
          value= { confirmPassword }
          onChange = { e => setConfirmPassword(e.target.value) }
        />
      </FormItem>
      <FormItem>
      <Button type="primary" htmlType="submit" style={{marginRight: '10px'}}>
          Signup
      </Button>
      Or 
      <NavLink 
          style={{marginRight: '10px'}} 
          to='/login'> login as a teacher
      </NavLink>
      </FormItem>
      <p style={{color: 'red'}}>{errorMSg}</p>
    </Form>
  );
}

export default RegistrationForm;