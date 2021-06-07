import React, { useState } from "react";
import { Form, Icon, Input, 
  Button, Spin, message}  from 'antd';
import { useHistory, useLocation } from "react-router-dom";
import {connect} from "react-redux"
import {NavLink, Redirect} from "react-router-dom";
import * as actions from "../store/actions/authActions"
import axios from "axios";
import LocalStorage from 'local-storage'

const FormItem = Form.Item;

const  NormalLoginForm = () => {

  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [errorMsg, setErrorMsg] = useState(false)

  const history = useHistory();
  
  const handleSubmit = (e) => { 
    e.preventDefault();
    axios.post('http://127.0.0.1:8000/login', {
        username: username,
        password: password
    }).then(res => {
        setErrorMsg(false)
        console.log('RES', res.data.user_name) 
        LocalStorage.set('auth', true)
        LocalStorage.set('username', res.data.user_name)
        history.push(`/savedtests/` + username + '/')
    }).catch (e => {
      setErrorMsg(true)
      console.log('error', e)
    })
  }
  //   let user_type = ''
  //   this.props.form.validateFields((err, values) => {
  //     if (!err) {
  //       this.props.onAuth(values.userName, values.password)
  //       axios.get(`http://127.0.0.1:8000/api/users`)
  //       .then(res => {
  //           let found = false
  //           {res.data.map(function(item, i){
  //             if(item.username == values.userName)
  //             {
  //               found = true
  //               user_type = item.user_type 
  //             }
  //           })}
  //           if(found)
  //           {
  //             this.setState({redirect: true, username: values.userName})
  //           }
  //           console.log(res.data) 
  //           console.log(user_type)
  //       })        
  //     }
  //     else
  //     {
  //       message.error('You have not entered anything, please try again!');
  //     }
  //   });
  // }
  
    // let errorMessage = null;
    // if(this.props.error){
    //   errorMessage = (
    //     <p>{" Your details were incorect, please try to login again!"}</p>
    //   )
    // }
    // if(this.state.redirect)
    // {
    //   return <Redirect to={`/modulesList/` + this.state.username + '/'}/>
    // }
    // const { getFieldDecorator } = this.props.form;
    return (
      <div>
        {/* {errorMessage} */}
        {
          // this.props.loading ?

          // <Spin indicator={antIcon} />
          
          // :

          <Form onSubmit={handleSubmit} className="login-form">

            <FormItem>
              {/* {getFieldDecorator('userName', {
                rules: [
                  {required: true, message: 'Please input your username!'},
                  {validator: this.checkUsername}
                ],
              })( */}
                <Input 
                  prefix={<Icon type="user"/>}  s
                  style={{ fontSize: 13 }} 
                  placeholder="Username" 
                  value={ username }
                  onChange= { e => setUsername(e.target.value)}  
                />
                {/* )} */}
            </FormItem>

            <FormItem>
              {/* {getFieldDecorator('password', {
                rules: [{ required: true, message: 'Please input your Password!' }],
              })( */}
                <Input 
                  prefix={<Icon type="lock" />} 
                  style={{ fontSize: 13 }} 
                  value={ password }
                  onChange= { e => setPassword(e.target.value)}
                  type="password" 
                  placeholder="Password" />
                {/* )} */}
            </FormItem>
            
            <FormItem>
                <Button type="primary" htmlType="submit" style={{marginRight: '12px'}}>login</Button>
                or
                <NavLink 
                  style={{marginRight: '11px'}} 
                  to="/registerteacher/"> Sign Up As a teacher
                </NavLink>
            </FormItem>
          </Form>
          
        }
        { errorMsg ? <p style={{ color: 'red'}}>Invalid username or password</p> : '' }
        
      </div>
    );
  
}

export default NormalLoginForm;

