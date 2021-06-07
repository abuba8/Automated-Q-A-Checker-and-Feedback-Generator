import React from "react";
import axios from "axios";
import "../css/ModuleList.css"
import {Link} from "react-router-dom";
import {Button, Table, Divider} from "antd";
class TestList extends React.Component {
    constructor(props)
    {
        super(props)
        this.state = {
            questions: [],
            module: [],
            entertest: false,
            question_id: 0,
            answer: null,
            showingAlert: false,
        }
        this.sendAnswer = this.sendAnswer.bind(this);
        this.handleChange = this.handleChange.bind(this);
        this.handleClick = this.handleClick.bind(this);
        this.handleDelete = this.handleDelete.bind(this);
    }
    
    componentDidMount()
    {
        axios.all([
            axios.get('http://127.0.0.1:8000/api/test'),
            axios.get('http://127.0.0.1:8000/api/modules')
        ])
        .then(axios.spread((questionres, moduleres) => {
                this.setState({
                    questions: questionres.data,
                    module: moduleres.data,
                    questionId: questionres.data.module
                })
                console.log(this.state.questions)
        }))
    }
    handleClick = () => {
        this.setState({
            entertest: true,
        })
    }
    handleDelete = (event) => {
        const questionId = this.props.match.params;
        console.log(questionId.id);
        
        console.log("wertf")
        axios.delete(`http://127.0.0.1:8000/api/test/${event}/delete`);
        
        this.props.history.push('/');
        this.forceUpdate();
    }
    handleChange = ({target}) => {
      this.setState({
        [target.name]: target.value
      })
    }
    sendAnswer(event) {
      console.log(this.refs.answer.value)
      axios.post(`http://127.0.0.1:8000/api/create/answer/`, {
        answer: this.refs.answer.value
      })
    }
      //saves the test to a Database with the teacher also saved!
    saveTest(event1, event2) {
      console.log("gh")
      console.log(event1 + "" + event2)
      axios.post(`http://127.0.0.1:8000/api/save/test/`,{
        test: event1,
        username: event2
      })
      .then(res => {
        this.setState({
          showingAlert: true
        });
        setTimeout(() => {
          this.setState({
            showingAlert: false,
          });
        }, 5000);
      })
    }
    render() 
    {
        var options = { year: 'numeric', month: 'long', day: 'numeric' };
        const {id} = this.props.match.params
        console.log(id)
        let title = ''
        let name = ''
        let questiontype = ''
        let question_id = 0
        let module_id = 0
        let date = ''
        let count = 0
        let student_num = 0
        let description = ''
            //collect the questions set if there is any
        {this.state.questions.map(function(item, i){
            if(item.module == id)
            {
                questiontype = item.questiontype
                question_id = item.id
                date = item.created_date
                count = item.test_count
                name = item.name
                description = item.description
            }
        })}
            //collect the title from the modules api
        {this.state.module.map(function(item, i){
            if(item.id == id)
            {
                module_id = item.id
                title = item.title
                student_num = item.num_students
            }
        })}
        
        const columns = [{
          title: 'Test Name',
          dataIndex: 'question',
          key: 'question',
          render: (text) => {
                      //    console.log(text);
            return {
              children: text,
              props: {
                'data-tip': 'a very long text',
              },
            };
          },
        }, {
          title: 'Test Count',
          dataIndex: 'test_count',
          key: 'test_count',
        },{
          title: 'Test Description',
          dataIndex: 'description',
          key: 'description',
        }, {
          title: 'Type of question',
          dataIndex: 'question_type',
          key: 'question_type',
        },{
          title: 'Date creation',
          dataIndex: 'date_creation',
          key: 'date_creation',
        },{
          title: 'Action',
          key: 'action',
          render: (text, record) => (
            <span>
              <Button onClick={(e) => this.saveTest(question_id, this.props.match.params.userid)} type="primary" htmlType="submit">Save Test</Button>
              <Divider type="vertical" />
              <Link to={`/addGrade/` + question_id + '/' + title + '/' + record.test_count}>
                  <Button type="primary" htmlType="submit">Grade Test</Button>
              </Link>
              <Divider type="vertical" />
              <Link to={`/questions/` + question_id + '/' + module_id + '/' + this.props.match.params.userid}>
                  <Button type="primary" htmlType="submit">Update</Button>
              </Link>
            </span>
          ),
        }];
          const data = [
            {
              key: '1',
              question: name,
              test_count: count,
              description: description,
              question_type: questiontype,
              date_creation: new Date(date).toLocaleDateString([], options),
            },
          ];
        console.log(this.state.module[0])
            return(
                <div >
                    <div className={`alert alert-success ${this.state.showingAlert ? 'alert-shown': 'alert-hidden'}`}>
                        <strong>The test for {title} has been saved!</strong> Go check your saved tests!!
                    </div>
                    <h1 style={{textAlign: 'center'}}>This is a test you created for {title} </h1>
                    <Table columns={columns} dataSource={data} />
                    <br />
                    
                </div>    
            )
    }
}
export default TestList;