import { Form, Input, Button, Row, Col, message, Card, Statistic} from 'antd';
import React from "react"
import axios from "axios";
import GradeMark from "./GeneralInput"
class GradeForm extends React.Component {
  constructor(props){
    super(props);
    this.state = {
      type: "default",
      num: 0,
      module: [],
      showingAlert: false,
      test: [],
      num_subquestions: 0,
      topicsAsked: false,
      topicSet:false,
      counter: 0,
      counterTwo: 0,
      topics: [],
      marksForTopicsAsked: false,
      markSet: false,
      marksForTopics: [],
      value: '',
      start: false,
      totalMark: 0,
      totalSub: 0
    }
    this.handleChange = this.handleChange.bind(this);
  }
  onChange = value => {
    this.setState({ value });
  };
  handleChange = (e) => {
    const val = e.label
    console.log(val)
    this.setState({
      type: e.label
    })
  }
  componentDidMount()
    {
        const TESTID = this.props.testID
        axios.get(`http://127.0.0.1:8000/api/test/${TESTID.id}`)
        .then(res => {
          this.setState({
            test: res.data,
            num_subquestions: res.data.num_subquestions,
            start: true
          })
        })
        
    }
        //request type given because form is used more than once
        //also specifying the module id for updating a specific module.
  handleFormSubmit = (e, requestMethod, testID) => {
      e.preventDefault();
      let grade = " ";
      let effectiveness = "";
      const total = this.state.totalMark;
      const total_sub = this.state.totalSub;
      const final_mark = parseInt((total_sub/total)*100)
      //calculate the weakest topic!
      let lowest = this.state.marksForTopics;
      let lowestIndex = 0;
      const minimum_mark = Math.min(...lowest)
      for(var i = 0; i < lowest.length; i++)
      {
        if(lowest[i] == minimum_mark)
        {
          lowestIndex = i;
        }
      }
      let weakest = ""
      this.state.topics.map(function(item, i){
        if(lowestIndex == i)
        {
          weakest = item
        }
      })
      console.log("the lowest topic is: " + weakest)
      
      if(final_mark >= 70 && final_mark <= 100)
      {
        grade = "A";
        effectiveness = "Outstanding";
      }
      else if(final_mark >= 60 && final_mark <= 69)
      {
        grade = "B";
        effectiveness = "Good";
      }
      else if(final_mark >= 50 && final_mark <= 59)
      {
        grade = "C";
        effectiveness = "Need Improvement";
      }
      else if(final_mark >= 45 && final_mark <= 49)
      {
        grade = "D";
        effectiveness = "Poor";
      }
      else
      {
        grade = "Fail";
        effectiveness = "Fail";
      }
      console.log(testID.id + " " + total + " " + total_sub + " " + final_mark)
      switch(requestMethod) {
        case 'post':
          axios.post(`http://127.0.0.1:8000/api/create/grade/`, {
            grade: grade,
            grade_mark: final_mark,
            effectiveness: effectiveness,
            test: testID.id
          })
          .then(res => {
            console.log(res) 
            let new_str = "";
            this.state.topics.map(function(item, i){
              if(i == 0)
              {
                new_str = new_str + item
              }
              else
              {
                new_str = new_str +","+ item
              }
            })
            let mark_str = "";
            this.state.marksForTopics.map(function(item, i){
              if(i == 0)
              {
                mark_str = mark_str + item
              }
              else
              {
                mark_str = mark_str + "," + item
              }  
            })
            console.log(new_str)
            axios.post('http://127.0.0.1:8000/api/create/answer/', {
              test: testID.id,
              total_mark_for_question: total,
              topics: new_str,
              total_sub_marks: total_sub,
              topic_mark_breakdown: mark_str,
              weakest_topic: weakest,
            })
            .then(res => {
              console.log(res)
            })
            this.setState({
                showingAlert: true
            });
            setTimeout(() => {
                this.setState({
                  showingAlert: false,
                });
            }, 5000);
          })
          .catch(err => console.log(err))
        case 'put':
            return null
      }
  }
  onKeyPress(event) {
    const keyCode = event.keyCode || event.which;
    const keyValue = String.fromCharCode(keyCode);
     if (/\+|-/.test(keyValue))
       event.preventDefault();
  }
  handleTopicSubmit(e, counter)
  {
    
    counter = counter+1;
    console.log(counter)
    e.preventDefault();
    const topic = e.target.elements.topic.value;
    message.success('We have successfully saved the topic: ' + topic);
    this.setState({
      counter: counter
    })
    
    var join = this.state.topics.concat(topic);
    this.setState({
      topics: join,
      topicSet: true
    })
    if(counter == this.state.num_subquestions)
    {
      this.setState({
        topicsAsked: true,
      })
    }
  }
  handleMarkForTopicSubmit(e, counterTwo)
  {
    
    counterTwo = counterTwo+1;
    console.log(counterTwo)
    e.preventDefault();
    const mark = e.target.elements.amount.value;
    message.success('We have successfully saved ' + mark + ' !.' );
    this.setState({
      counterTwo: counterTwo
    })
    
    var join = this.state.marksForTopics.concat(mark);
    this.setState({
      marksForTopics: join,
      markSet: true
    })
    console.log(this.state.num_subquestions)
    if(counterTwo == this.state.num_subquestions)
    {
      console.log("nf")
      this.setState({
        marksForTopicsAsked: true,
      })
    }
  }
  handleTotalSubmit(e){
    const total = e.target.elements.total_mark.value;
    const total_sub = e.target.elements.total_sub.value;
    this.setState({
      start: false,
      totalMark: total,
      totalSub: total_sub
    })
  }
  render() {
    let list_topics = []
    let list_marks = []
    this.state.topics.map(function(item, i){
      list_topics.push((i+1) + ".) " + item)
    })
    this.state.marksForTopics.map(function(item, i){
      list_marks.push((i+1) + ".) " + item)
    })
    console.log(this.state.marksForTopicsAsked)
    return (
      <div>
        <div className={`alert alert-success ${this.state.showingAlert ? 'alert-shown': 'alert-hidden'}`}>
                <strong>The grade that you have given for this test has been saved!</strong>Go check your saved tests to check them out!!
        </div>
        {
          this.state.start ?
          <Form style={{textAlign: 'center'}} onSubmit={(event) => this.handleTotalSubmit(event)}>
            <Row gutter={20}>
              <Col style={{textAlign: 'center', marginLeft: '32%'}} span={8}>
                <Form.Item label="Total mark for all questions">
                  <Input type="number" name="total_mark" pattern="[0-9]*" onKey Press={this.onKeyPress.bind(this)} />
                </Form.Item>
                <Form.Item label="Total mark given to students for sub questions">
                    <Input type="number" name="total_sub" pattern="[0-9]*" onKey Press={this.onKeyPress.bind(this)} />
                </Form.Item>
              </Col>
            </Row>
          <Form.Item>
            <Button type="primary" htmlType="submit">Add information</Button>
            </Form.Item>
          </Form>
          :
          this.state.start == false
        }
        
        {
          this.state.start == false ?

          this.state.counterTwo != this.state.num_subquestions && this.state.topicsAsked ?
          <Form onSubmit={(event) => this.handleMarkForTopicSubmit(event,this.state.counterTwo)}>
            <Form.Item>
              <Row>
              <Card bordered style={{color: 'blue', textAlign: 'center', fontSize: '10px'}} title="Topic reminder" bordered={false}>
                {list_topics.map(options => (
                    <Col span={12}>
                            <Statistic title={" "} value={options} suffix={" "} />
                    </Col>
                ))}
              </Card>
              </Row>
            </Form.Item>
            <div style={{textAlign: 'center'}}>
                <strong>All of your topics were saved! Now add the marks for these topics</strong>
            </div>
            <Row>
              <Col>
              <Form.Item style={{textAlign: 'center'}} label={"What was the mark for topic " + (this.state.counterTwo+1) + " out of " + this.state.totalSub}>
                    <GradeMark name="amount" style={{ width: 120 }} value={this.state.value} onChange={this.onChange} />
                </Form.Item>
                <Form.Item style={{textAlign: 'center'}}>
                    <Button type="primary" htmlType="submit">Add Mark</Button>
                </Form.Item>
              </Col>
            </Row>
          </Form>
          :
          this.state.counter != this.state.num_subquestions ?
          <Form onSubmit={(event) => this.handleTopicSubmit(event,this.state.counter)}>
            <Row>
              <Col>
                <Form.Item style={{textAlign: 'center'}} label={"What was the topic for question " + (this.state.counter+1)}>
                  <Input style={{width: '150px'}}name="topic" placeholder="" />
                </Form.Item>
                <Form.Item style={{textAlign: 'center'}}>
                    <Button type="primary" htmlType="submit">Add topic</Button>
                </Form.Item>
              </Col>
            </Row>
          </Form>
          
          :
          
          <Form onSubmit={(event) => this.handleFormSubmit(event,this.props.requestMethod, this.props.testID)}>
            <h3 style={{textAlign: 'center'}}>Well done all of your topics were saved!</h3>
            <h3 style={{textAlign: 'center'}}>Total number of of marks: <b style={{color: 'green'}}>{this.state.totalMark}</b></h3>
            <h3 style={{textAlign: 'center'}}>Total sub marks: <b style={{color: 'green'}}>{this.state.totalSub}</b></h3>
            <Form.Item>
              <Row>
              <Card bordered style={{color: 'blue', textAlign: 'center', fontSize: '10px'}} title="Topics" bordered={false}>
                {list_topics.map(options => (
                    <Col span={12}>
                            <Statistic title={" "} value={options} suffix={" "} />
                    </Col>
                ))}
                
              </Card>
              <Card bordered style={{color: 'blue', textAlign: 'center', fontSize: '10px'}} title="Marks" bordered={false}>
                
                {list_marks.map(options => (
                    <Col span={12}>
                            <Statistic title={" "} value={options} suffix={" "} />
                    </Col>
                ))}
              </Card>
              </Row>
            </Form.Item>
            <Form.Item style={{textAlign: 'center'}}>
              <Button type="primary" htmlType="submit">{this.props.btnText}</Button>
            </Form.Item>
          </Form>
          :
          null
        }
        
      </div>
    );
  }
}
export default GradeForm